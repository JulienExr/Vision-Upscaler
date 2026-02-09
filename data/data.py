import os
import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
from PIL import Image, ImageFilter
import numpy as np


class QualityEnhancementDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.transforms=transforms

    def __len__(self):
        return len(self.image_paths)
    
    def degrade_image(self, img):
        choice = random.choice(['noise', 'blur', 'pixelate', 'mixed'])
        img = Image(img)
        if choice == 'noise':
            img_np = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 25, img.shape())
            img_np = np.clip(img_np + noise, 0, 255). astype(np.uint8)
            img = Image.fromarray(img_np)
        
        elif choice == 'blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1,3)))
        
        elif choice == 'pixelate':
            img = img.resize((img.width // 4, img.height // 4), resample=Image.NEAREST)
            img = img.resize((img.width * 4, img.height * 4), resample=Image.NEAREST)
        
        elif choice == 'mixed':
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1,3)))
            img_np = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 25, img.shape())
            img_np = np.clip(img_np + noise, 0, 255). astype(np.uint8)
            img = Image.fromarray(img_np)
        
        return img
    

    def __getitem__(self, index):
        clean_img = Image.open(self.image_paths[index]).convert('RGB')
        degraded_img = self.degrade_image(clean_img)

        if self.transforms:
            clean_img = self.transforms(clean_img)
            degraded_img = self.transforms(degraded_img)

        return degraded_img, clean_img
    

def get_dataloader(directory, batch_size=32, num_workers=4):
    
    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f'**/*{ext.lower()}'), recursive=True))
        image_paths.extend(glob.glob(os.path.join(directory, f'**/*{ext.upper()}'), recursive=True))

    image_paths = list(set(image_paths))
    
    random.shuffle(image_paths)

    idx_split = int(0.8 * len(image_paths))
    train_paths = image_paths[:idx_split]
    val_paths = image_paths[idx_split:]

    train_dataset = QualityEnhancementDataset(train_paths, transforms=transforms)
    val_dataset = QualityEnhancementDataset(val_paths, transforms=transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader