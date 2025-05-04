import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from utils.dataloader import My_test_dataset
from lib.mamba_unet import CamouflageDetectionNet


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='/kaggle/input/avgnet/other/default/1/Net_epoch_best.pth')
parser.add_argument('--test_path', type=str,default='/kaggle/input/cottonworm4/CottonWorm4_Drive',help='path to testing dataset')


opt = parser.parse_args()

for _data_name in [opt.test_path]:
    data_path = f'{_data_name}/test'
    save_path = './results/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    
    model = CamouflageDetectionNet(pretrained=True).cuda()

    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root',image_root,gt_root)
    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    print('****',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name',name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        P1, P2 = model(image)

        os.makedirs(save_path+"/out1out2out3out4Final", exist_ok=True)
        res = F.upsample(P1[0] + P1[1] + P1[2] + P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out1out2out3out4Final/"+name,res*255)

        os.makedirs(save_path+"/out2out3out4Final", exist_ok=True)
        res = F.upsample(P1[1] + P1[2] + P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out2out3out4Final/"+name,res*255)

        os.makedirs(save_path+"/out3out4Final", exist_ok=True)
        res = F.upsample(P1[2] + P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out3out4Final/"+name,res*255)

        os.makedirs(save_path+"/out4Final", exist_ok=True)
        res = F.upsample(P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out4Final/"+name,res*255)

        os.makedirs(save_path+"/final", exist_ok=True)
        res = F.upsample(P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/final/"+name,res*255)
        
        os.makedirs(save_path+"/out1", exist_ok=True)
        res = F.upsample(P1[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out1/"+name,res*255)

        os.makedirs(save_path+"/out2out4final", exist_ok=True)
        res = F.upsample(P1[1] + P1[-1] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out2out4final/"+name,res*255)
