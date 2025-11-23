import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from utils.dataloader import My_test_dataset
from lib.mamba_unet import CamouflageDetectionNet

import matplotlib.pyplot as plt

def generate_gradcam(model, image_tensor, target_layer, class_idx=None):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Registra hooks
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # Forward
    outputs, final_output = model(image_tensor)
    if class_idx is None:
        class_idx = final_output.argmax(dim=1).item() if final_output.shape[1] > 1 else 0

    # Backward
    model.zero_grad()
    class_score = final_output[0, class_idx].sum()
    class_score.backward()

    grads_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations_val[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = np.uint8(cam * 255)
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))

    # Limpia los hooks
    handle_fw.remove()
    handle_bw.remove()

    return cam

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='/kaggle/input/avgnet/other/default/1/Net_epoch_best.pth')
parser.add_argument('--test_path', type=str,default='/kaggle/input/cottonworm4/CottonWorm4_Drive',help='path to testing dataset')


opt = parser.parse_args()

for _data_name in [opt.test_path]:
    data_path = f'{_data_name}/test'
    save_path = './results/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    
    model = CamouflageDetectionNet(pretrained=True).cuda()

    # Recorre todas las subcapas y muestra las convolucionales
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name}: {module}")

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

        os.makedirs(save_path+"/final", exist_ok=True)
        res = F.upsample(P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/final/"+name,res*255)

        os.makedirs(save_path+"/out4final", exist_ok=True)
        res = F.upsample(P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out4final/"+name,res*255)

        os.makedirs(save_path+"/out3out4final", exist_ok=True)
        res = F.upsample(P1[2] + P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out3out4final/"+name,res*255)

        os.makedirs(save_path+"/out2out3out4final", exist_ok=True)
        res = F.upsample(P1[1] + P1[2] + P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out2out3out4final/"+name,res*255)

        os.makedirs(save_path+"/out1out2out3out4final", exist_ok=True)
        res = F.upsample(P1[0] + P1[1] + P1[2] + P1[3] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+"/out1out2out3out4final/"+name,res*255)

        # Grad-CAM 1
        # Elige la capa: la última capa convolucional del final_decoder
        target_layer = model.final_decoder.res_block1.conv  # O ajusta según tu modelo
    
        # image: tensor [1, 3, H, W]
        cam = generate_gradcam(model, image, target_layer)
    
        # Prepara la imagen de entrada para superponer
        img_np = image.cpu().squeeze().permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
        # Superpone el Grad-CAM
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
        # Guarda el resultado
        os.makedirs(save_path+"/gradcam_1", exist_ok=True)
        cv2.imwrite(save_path+"/gradcam_1/"+name, superimposed_img)
        print('> Grad-CAM guardado en', save_path+"/gradcam/"+name)
        
        # Grad-CAM 2
        # Elige la capa: la última capa convolucional del final_decoder
        target_layer = model.final_decoder.res_block2.conv  # O ajusta según tu modelo
    
        # image: tensor [1, 3, H, W]
        cam = generate_gradcam(model, image, target_layer)
    
        # Prepara la imagen de entrada para superponer
        img_np = image.cpu().squeeze().permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
        # Superpone el Grad-CAM
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
        # Guarda el resultado
        os.makedirs(save_path+"/gradcam_2", exist_ok=True)
        cv2.imwrite(save_path+"/gradcam_2/"+name, superimposed_img)
        print('> Grad-CAM guardado en', save_path+"/gradcam/"+name)

        # Grad-CAM D1
        # Elige la capa: la última capa convolucional del final_decoder
        target_layer = model.decoder1.umamba_block.res_block2.conv  # O ajusta según tu modelo
    
        # image: tensor [1, 3, H, W]
        cam = generate_gradcam(model, image, target_layer)
    
        # Prepara la imagen de entrada para superponer
        img_np = image.cpu().squeeze().permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
        # Superpone el Grad-CAM
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
        # Guarda el resultado
        os.makedirs(save_path+"/gradcam_d1", exist_ok=True)
        cv2.imwrite(save_path+"/gradcam_d1/"+name, superimposed_img)
        print('> Grad-CAM guardado en', save_path+"/gradcam/"+name)

        # Grad-CAM D2
        # Elige la capa: la última capa convolucional del final_decoder
        target_layer = model.decoder2.umamba_block.res_block2.conv  # O ajusta según tu modelo
    
        # image: tensor [1, 3, H, W]
        cam = generate_gradcam(model, image, target_layer)
    
        # Prepara la imagen de entrada para superponer
        img_np = image.cpu().squeeze().permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
        # Superpone el Grad-CAM
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
        # Guarda el resultado
        os.makedirs(save_path+"/gradcam_d2", exist_ok=True)
        cv2.imwrite(save_path+"/gradcam_d2/"+name, superimposed_img)
        print('> Grad-CAM guardado en', save_path+"/gradcam/"+name)

        # Grad-CAM seg_heads.0
        # Elige la capa: la última capa convolucional del final_decoder
        target_layer = model.seg_heads.0  # O ajusta según tu modelo
    
        # image: tensor [1, 3, H, W]
        cam = generate_gradcam(model, image, target_layer)
    
        # Prepara la imagen de entrada para superponer
        img_np = image.cpu().squeeze().permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
        # Superpone el Grad-CAM
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
        # Guarda el resultado
        os.makedirs(save_path+"/gradcam_seg_0", exist_ok=True)
        cv2.imwrite(save_path+"/gradcam_seg_0/"+name, superimposed_img)
        print('> Grad-CAM guardado en', save_path+"/gradcam/"+name)

        # Grad-CAM CBAM
        # Elige la capa: la última capa convolucional del final_decoder
        target_layer = model.final_decoder.res_block2.conv  # O ajusta según tu modelo
    
        # image: tensor [1, 3, H, W]
        cam = generate_gradcam(model, image, target_layer)
    
        # Prepara la imagen de entrada para superponer
        img_np = image.cpu().squeeze().permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
        # Superpone el Grad-CAM
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
        # Guarda el resultado
        os.makedirs(save_path+"/gradcam_cbam", exist_ok=True)
        cv2.imwrite(save_path+"/gradcam_cbam/"+name, superimposed_img)
        print('> Grad-CAM guardado en', save_path+"/gradcam/"+name)


        
