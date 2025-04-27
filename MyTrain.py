import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from lib.mamba_unet import CamouflageDetectionNet  # Añade esta línea
from torch import optim

#delaited convolution
#lovasz_hinge

####
####CUDA_VISIBLE_DEVICES=0 python3 Train.py
####
def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def flatten_binary_scores(scores, labels):
    """
    Flattens predictions and labels for binary classification.
    Removes pixels with label -1 (optional ignore label).
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    valid = (labels != -1)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
    
def lovasz_hinge(logits, labels, per_image=True):
    """
    Binary Lovasz hinge loss
    logits: [B, 1, H, W] (raw predictions, no sigmoid)
    labels: [B, 1, H, W] (binary ground truth: 0 or 1)
    """
    if per_image:
        loss = torch.mean(torch.stack([
            lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0)))
            for log, lab in zip(logits, labels)
        ]))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Flat binary Lovasz hinge loss
    logits: [P] Variable, logits at each prediction (between -∞ and +∞)
    labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        return logits.sum() * 0.

    signs = 2. * labels - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]

    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    gt_sorted: sorted ground truth labels (descending)
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum().item()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
    
def structure_loss2(pred, mask):
    # Weighted BCE
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted IoU
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # Lovasz hinge loss
    lovasz = lovasz_hinge(pred, mask)

    return (wbce + wiou).mean() + lovasz
    
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def val(model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        test_loader = test_dataset(image_root=opt.test_path + '/Imgs/',
                                   gt_root=opt.test_path + '/GT/',
                                   testsize=opt.trainsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, res1 = model(image)
            combined_res = res[1] + res[2] + res[3] + res1 

            # eval Dice
            res = F.interpolate(combined_res, size=gt.shape[-2:], mode='bilinear', align_corners=False) # Usar gt.shape[-2:] para obtener H, W
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info('[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [1]
    loss_P1_record = AvgMeter()
    loss_P2_record = AvgMeter()
        
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # print('this is i',i)
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            #print('this is trainsize',trainsize)
            P1, P2 = model(images)
            # ---- loss function ----
            losses = [structure_loss(out, gts) for out in P1]
            loss_P1 = 0
            gamma = 0.25
            #print('iteration num',len(P1))
            for it in range(len(P1)):
                loss_P1 += (gamma * it) * losses[it]

            loss_P2 = structure_loss(P2, gts)

            loss = loss_P1 + loss_P2
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)
                loss_P2_record.update(loss_P2.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss P1: [{:0.4f}], Loss P2: [{:0.4f}]'.format(datetime.now(), epoch, opt.epoch, i, total_step,loss_P1_record.show(), loss_P2_record.show()))
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss P1: [{:0.4f}], Loss P2: [{:0.4f}]'.format(datetime.now(), epoch, opt.epoch, i, total_step,loss_P1_record.show(), loss_P2_record.show()))
    # save model
    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), save_path + str(epoch) + 'AIVGNet-PVT.pth')
    
if __name__ == '__main__':

    ##################model_name#############################
    #dataset = 'CottonWorm4_Drive'
    dataset = '/kaggle/input/cottonworm4/CottonWorm4_Drive'

    ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,default=1001, help='epoch number')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',default=True, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,default=352, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--decay_rate', type=float,default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,default=30, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,default=f'{dataset}/train',help='path to train dataset')
    parser.add_argument('--test_path', type=str,default=f'{dataset}/val',help='path to testing dataset')
    parser.add_argument('--save_path', type=str,default=f'./model_pth/AIVGNet_{dataset}/')
    parser.add_argument('--epoch_save', type=int,default=5, help='every n epochs to save model')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    logging.basicConfig(filename=opt.save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = CamouflageDetectionNet(pretrained=True).cuda()

    if opt.load is not None:
        pretrained_dict=torch.load(opt.load)
        print('!!!!!!Sucefully load model from!!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('model paramters',sum(p.numel() for p in model.parameters() if p.requires_grad))

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    #print(optimizer)
    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)

    print("image_root: ", image_root)
    print("gt_root: ", gt_root)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)

    writer = SummaryWriter(opt.save_path + 'summary')

    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1
    best_epoch = 0
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)

    for epoch in range(1, opt.epoch):
         # schedule
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch) 
        
        # train
        train(train_loader, model, optimizer, epoch, opt.save_path)
        if epoch % opt.epoch_save==0:
            # validation
            val(model, epoch, opt.save_path, writer)
