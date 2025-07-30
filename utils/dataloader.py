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

        # Contar imágenes originales
        original_images = [f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        original_gts = [f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        print(f"📊 ESTADÍSTICAS DEL DATASET:")
        print(f"   • Imágenes originales encontradas: {len(original_images)}")
        print(f"   • Ground truths originales encontrados: {len(original_gts)}")

        self.images = sorted([os.path.join(image_root, f) for f in original_images])
        self.gts = sorted([os.path.join(gt_root, f) for f in original_gts])

        # Preprocesar y redimensionar UNA SOLA VEZ
        self.preprocess_and_resize()
        self.size = len(self.processed_data)

        print(f"   • Pares procesados exitosamente: {self.size}")

        # Mostrar información sobre augmentaciones
        if self.augmentations:
            print(f"   • 🔄 AUGMENTACIONES ACTIVADAS:")
            print(f"     - Cada imagen se transforma aleatoriamente en cada época")
            print(f"     - Variaciones efectivas por época: {self.size} × múltiples transformaciones")
            print(f"     - HorizontalFlip, VerticalFlip, Rotaciones, Brillo, Contraste, etc.")
        else:
            print(f"   • ❌ AUGMENTACIONES DESACTIVADAS:")
            print(f"     - Solo resize y normalización")
            print(f"     - Número fijo de imágenes: {self.size}")

        # Configurar transformaciones
        self.transform = self.get_transforms()

    def preprocess_and_resize(self):
        """
        Preprocesa y redimensiona las imágenes UNA SOLA VEZ al inicializar
        """
        print(f"\n🔧 PREPROCESANDO IMÁGENES (una sola vez):")

        self.processed_data = []
        resized_count = 0
        error_count = 0

        for img_path, gt_path in zip(self.images, self.gts):
            try:
                # Cargar imagen y máscara
                image = Image.open(img_path).convert('RGB')
                gt = Image.open(gt_path).convert('L')

                # Verificar si necesitan resize
                if image.size != gt.size:
                    resized_count += 1
                    target_size = image.size  # Usar tamaño de la imagen como referencia
                    print(f"   📏 Redimensionando {os.path.basename(gt_path)}: {gt.size} → {target_size}")

                    # Resize de la máscara usando interpolación bicúbica
                    gt = gt.resize(target_size, Image.BICUBIC)

                # Convertir a numpy arrays y guardar
                image_np = np.array(image)
                gt_np = np.array(gt)

                self.processed_data.append((image_np, gt_np))

            except Exception as e:
                error_count += 1
                print(f"   ❌ Error procesando {os.path.basename(img_path)}: {str(e)}")

        print(f"\n📈 RESUMEN DEL PREPROCESAMIENTO:")
        print(f"   • Pares procesados: {len(self.processed_data)}")
        print(f"   • Máscaras redimensionadas: {resized_count}")
        print(f"   • Errores encontrados: {error_count}")
        print(f"   • ✅ Todas las imágenes están listas para augmentación")

    def get_transforms(self):
        if self.augmentations:
            print('\n🎨 CONFIGURANDO AUGMENTACIONES AVANZADAS:')
            transform = A.Compose([
                A.Resize(self.trainsize, self.trainsize),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=45, 
                    p=0.5, 
                    border_mode=0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])

            print('   ✅ Augmentaciones configuradas:')
            print('      - Geométricas: Flip H/V (50%), Rotación 90° (50%), ShiftScaleRotate (50%)')
            print('      - Fotométricas: Brillo/Contraste (50%), Ruido Gaussiano (30%)')
            print('      - Color: HSV (30%)')
            print('      - Normalización ImageNet + ToTensor')

            return transform
        else:
            print('\n➡️  CONFIGURANDO TRANSFORMACIONES BÁSICAS:')
            transform = A.Compose([
                A.Resize(self.trainsize, self.trainsize),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])
            print('   ✅ Solo resize + normalización configurados')
            return transform

    def __getitem__(self, index):
        # Obtener datos preprocesados (ya redimensionados una vez)
        image_np, gt_np = self.processed_data[index]

        # Aplicar augmentaciones (si están activadas)
        augmented = self.transform(image=image_np, mask=gt_np)
        image = augmented['image']
        gt = augmented['mask'].unsqueeze(0).float() / 255.0

        return image, gt

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=True):
    print(f"\n🚀 CREANDO DATALOADER:")
    print(f"   • Directorio de imágenes: {image_root}")
    print(f"   • Directorio de ground truths: {gt_root}")
    print(f"   • Tamaño de entrenamiento: {trainsize}x{trainsize}")
    print(f"   • Batch size: {batchsize}")
    print(f"   • Augmentaciones: {'✅ Activadas' if augmentation else '❌ Desactivadas'}")

    dataset = CODataset(image_root, gt_root, trainsize, augmentation)

    # Calcular información adicional del entrenamiento
    total_batches = len(dataset) // batchsize
    remaining_samples = len(dataset) % batchsize

    print(f"\n📈 INFORMACIÓN DE ENTRENAMIENTO:")
    print(f"   • Total de imágenes para entrenamiento: {len(dataset)}")
    print(f"   • Batches por época: {total_batches}")
    if remaining_samples > 0:
        print(f"   • Muestras en el último batch: {remaining_samples}")

    if augmentation:
        print(f"   • 🎲 Variaciones por época: INFINITAS (transformaciones aleatorias)")
        print(f"   • 🔄 Cada época verá versiones diferentes de las mismas imágenes")
    else:
        print(f"   • 📊 Muestras fijas por época: {len(dataset)}")

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
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
        print(f"   • ⚠️  NOTA: Las imágenes de prueba se redimensionan en cada load_data()")

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])

        # Verificar y ajustar tamaños si es necesario (solo para test)
        if image.size != gt.size:
            gt = gt.resize(image.size, Image.BICUBIC)

        image = self.transform(image).unsqueeze(0)

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

# Función para mostrar resumen completo del dataset
def show_dataset_summary(train_loader, test_dataset=None):
    """
    Muestra un resumen completo del dataset con información de augmentaciones
    """
    print(f"\n" + "="*70)
    print(f"📋 RESUMEN COMPLETO DEL DATASET")
    print(f"="*70)

    # Información del dataset de entrenamiento
    train_dataset = train_loader.dataset
    print(f"🏋️  ENTRENAMIENTO:")
    print(f"   • Imágenes base: {len(train_dataset)}")
    print(f"   • Batch size: {train_loader.batch_size}")
    print(f"   • Batches por época: {len(train_loader)}")
    print(f"   • Preprocesamiento: ✅ Una sola vez al inicializar")

    if train_dataset.augmentations:
        print(f"   • Augmentaciones: ✅ ACTIVADAS")
        print(f"     - Cada imagen se transforma aleatoriamente")
        print(f"     - Variaciones por época: INFINITAS")
        print(f"     - Diversidad: Muy alta")
    else:
        print(f"   • Augmentaciones: ❌ DESACTIVADAS")
        print(f"     - Imágenes fijas por época: {len(train_dataset)}")
        print(f"     - Diversidad: Limitada")

    if test_dataset:
        print(f"\n🧪 PRUEBA:")
        print(f"   • Imágenes de prueba: {len(test_dataset)}")
        print(f"   • Augmentaciones: ❌ Desactivadas (solo resize)")

    print(f"\n💾 CONFIGURACIÓN:")
    print(f"   • Tamaño de imagen: {train_dataset.trainsize}x{train_dataset.trainsize}")
    print(f"   • Shuffle: {'✅' if train_loader.sampler is None else '❌'}")
    print(f"   • Num workers: {train_loader.num_workers}")
    print(f"   • Pin memory: {'✅' if train_loader.pin_memory else '❌'}")
    print(f"="*70)

# Función para demostrar el efecto de las augmentaciones
def demonstrate_augmentations(dataset, num_samples=3):
    """
    Demuestra cómo las augmentaciones crean diferentes versiones de la misma imagen
    """
    if not dataset.augmentations:
        print("❌ Las augmentaciones están desactivadas")
        return

    print(f"\n🎨 DEMOSTRACIÓN DE AUGMENTACIONES:")
    print(f"Mostrando {num_samples} transformaciones de la primera imagen...")

    for i in range(num_samples):
        image, gt = dataset[0]  # Siempre la misma imagen base
        print(f"   Muestra {i+1}: Tensor shape {image.shape}, GT shape {gt.shape}")
        print(f"   - Valores únicos en imagen: {len(torch.unique(image))}")
        print(f"   - Min/Max imagen: {image.min():.3f}/{image.max():.3f}")

    print("✅ Cada llamada produce una transformación diferente!")

def verify_dataset_integrity(image_root, gt_root):
    """
    Verifica la integridad del dataset antes del preprocesamiento
    """
    print(f"\n🔍 VERIFICANDO INTEGRIDAD DEL DATASET...")

    images = sorted([f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    gts = sorted([f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    size_stats = {}
    mismatches = 0

    for img_name, gt_name in zip(images, gts):
        try:
            img = Image.open(os.path.join(image_root, img_name))
            gt = Image.open(os.path.join(gt_root, gt_name))

            img_size = img.size
            gt_size = gt.size

            if img_size != gt_size:
                mismatches += 1

            # Estadísticas de tamaños
            if img_size not in size_stats:
                size_stats[img_size] = 0
            size_stats[img_size] += 1

        except Exception as e:
            print(f"   ❌ Error con {img_name}: {e}")

    print(f"\n📊 ESTADÍSTICAS DE TAMAÑOS:")
    for size, count in sorted(size_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {size}: {count} imágenes")

    print(f"\n📈 RESUMEN:")
    print(f"   • Total de pares: {len(images)}")
    print(f"   • Pares con tamaños diferentes: {mismatches}")
    print(f"   • 🔧 Se redimensionarán {mismatches} máscaras UNA SOLA VEZ")

    return len(images), mismatches
