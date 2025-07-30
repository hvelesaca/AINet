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

        # Contar imágenes originales antes del filtrado
        original_images = [f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        original_gts = [f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        print(f"📊 ESTADÍSTICAS DEL DATASET:")
        print(f"   • Imágenes originales encontradas: {len(original_images)}")
        print(f"   • Ground truths originales encontrados: {len(original_gts)}")

        self.images = sorted([os.path.join(image_root, f) for f in original_images])
        self.gts = sorted([os.path.join(gt_root, f) for f in original_gts])

        # Filtrar archivos y mostrar estadísticas
        self.filter_files()
        self.size = len(self.images)

        print(f"   • Imágenes válidas después del filtrado: {self.size}")

        # Mostrar información sobre augmentaciones
        if self.augmentations:
            print(f"   • Augmentaciones ACTIVADAS - cada imagen se puede transformar de múltiples formas")
            print(f"   • Número efectivo de variaciones por época: {self.size} × (transformaciones aleatorias)")
        else:
            print(f"   • Augmentaciones DESACTIVADAS - solo resize y normalización")
            print(f"   • Número de imágenes para entrenamiento: {self.size}")

        self.transform = self.get_transforms()

    def get_transforms(self):
        if self.augmentations:
            print('🔄 Usando Albumentations avanzadas para augmentación:')
            print('   - HorizontalFlip (50%), VerticalFlip (50%), RandomRotate90 (50%)')
            print('   - ShiftScaleRotate (50%), RandomBrightnessContrast (50%)')
            print('   - GaussNoise (30%), HueSaturationValue (30%)')
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
            print('➡️  Sin augmentación, solo resize y normalización')
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
        assert len(self.images) == len(self.gts), f"❌ Error: {len(self.images)} imágenes vs {len(self.gts)} ground truths"

        images, gts = [], []
        filtered_count = 0

        for img_path, gt_path in zip(self.images, self.gts):
            try:
                img = Image.open(img_path)
                gt = Image.open(gt_path)

                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
                else:
                    filtered_count += 1
                    print(f"   ⚠️  Filtrada: {os.path.basename(img_path)} - tamaños no coinciden")
            except Exception as e:
                filtered_count += 1
                print(f"   ❌ Error al abrir: {os.path.basename(img_path)} - {str(e)}")

        if filtered_count > 0:
            print(f"   • Imágenes filtradas (eliminadas): {filtered_count}")

        self.images, self.gts = images, gts

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=True):
    print(f"\n🚀 CREANDO DATALOADER:")
    print(f"   • Directorio de imágenes: {image_root}")
    print(f"   • Directorio de ground truths: {gt_root}")
    print(f"   • Tamaño de entrenamiento: {trainsize}x{trainsize}")
    print(f"   • Batch size: {batchsize}")

    dataset = CODataset(image_root, gt_root, trainsize, augmentation)

    # Calcular información adicional del entrenamiento
    total_batches = len(dataset) // batchsize
    remaining_samples = len(dataset) % batchsize

    print(f"\n📈 INFORMACIÓN DE ENTRENAMIENTO:")
    print(f"   • Total de imágenes para entrenamiento: {len(dataset)}")
    print(f"   • Batches por época: {total_batches}")
    if remaining_samples > 0:
        print(f"   • Muestras en el último batch: {remaining_samples}")
    print(f"   • Muestras procesadas por época: {len(dataset)}")

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

        print(f"\n🧪 DATASET DE PRUEBA:")
        print(f"   • Imágenes de prueba: {len(self.images)}")
        print(f"   • Ground truths de prueba: {len(self.gts)}")

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

        print(f"\n🔍 MI DATASET DE PRUEBA:")
        print(f"   • Imágenes de prueba: {len(self.images)}")
        print(f"   • Ground truths de prueba: {len(self.gts)}")

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

    def __len__(self):
        return self.size

# Función adicional para mostrar resumen completo del dataset
def show_dataset_summary(train_loader, test_dataset=None):
    """
    Muestra un resumen completo del dataset
    """
    print(f"\n" + "="*60)
    print(f"📋 RESUMEN COMPLETO DEL DATASET")
    print(f"="*60)

    # Información del dataset de entrenamiento
    train_dataset = train_loader.dataset
    print(f"🏋️  ENTRENAMIENTO:")
    print(f"   • Imágenes totales: {len(train_dataset)}")
    print(f"   • Batch size: {train_loader.batch_size}")
    print(f"   • Batches por época: {len(train_loader)}")
    print(f"   • Augmentaciones: {'✅ Activadas' if train_dataset.augmentations else '❌ Desactivadas'}")

    if test_dataset:
        print(f"\n🧪 PRUEBA:")
        print(f"   • Imágenes de prueba: {len(test_dataset)}")

    print(f"\n💾 CONFIGURACIÓN:")
    print(f"   • Tamaño de imagen: {train_dataset.trainsize}x{train_dataset.trainsize}")
    print(f"   • Shuffle: {'✅' if train_loader.sampler is None else '❌'}")
    print(f"   • Num workers: {train_loader.num_workers}")
    print(f"   • Pin memory: {'✅' if train_loader.pin_memory else '❌'}")
    print(f"="*60)
