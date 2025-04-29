import os
from PIL import Image
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torch.utils.data as data
import albumentations as A
import numpy as np
import torch

class CODataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, augmentations=True):
        self.trainsize = trainsize
        self.augmentations = augmentations

        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        self.filter_files()
        self.size = len(self.images)

        self.transform = self.get_transforms()

    def get_transforms(self):
        if self.augmentations:
            print('Using advanced Albumentations for augmentation')
            return A.Compose([
                A.Resize(self.trainsize, self.trainsize),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=0),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],)
        else:
            print('No augmentation, only resizing and normalization')
            return A.Compose([
                A.Resize(self.trainsize, self.trainsize),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],)  


    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert('RGB'))
        gt = np.array(Image.open(self.gts[index]).convert('L'))
        
        augmented = self.transform(image=image, mask=gt)
        image = augmented['image']
        gt = augmented['mask'].unsqueeze(0).float() / 255.0
        
        return image, gt
        
    def filter_files(self):
        assert len(self.images) == len(self.gts), "Mismatch between images and ground truths"

        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            
            if img.size == gt.size == edg.size:
                images.append(img_path)
                gts.append(gt_path)
            
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=True):
    dataset = CODataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPG')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        # self.index += 1
        # self.index = self.index % self.size
        # return image, gt, name
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class My_test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
       
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
