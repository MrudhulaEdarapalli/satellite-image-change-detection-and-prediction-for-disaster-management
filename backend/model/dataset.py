import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class OneraDataset(Dataset):
    def __init__(self, root_dir, cities, patch_size=96, stride=48, transform=None):
        self.root_dir = root_dir
        self.cities = cities
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = self._make_patches()

    def _make_patches(self):
        patches = []
        for city in self.cities:
            img1_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Images", city, "pair", "img1.png")
            img2_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Images", city, "pair", "img2.png")
            
            # Check for label in Train Labels first, then Test Labels
            label_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Train Labels", city, "cm", "cm.png")
            if not os.path.exists(label_path):
                label_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Test Labels", city, "cm", "cm.png")

            if not os.path.exists(img1_path) or not os.path.exists(img2_path) or not os.path.exists(label_path):
                print(f"Skipping city {city} due to missing files.")
                continue

            with Image.open(img1_path) as img:
                w, h = img.size

            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    patches.append({
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'label_path': label_path,
                        'coords': (i, j)
                    })
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        i, j = patch_info['coords']
        
        # Load images and label (loading full image and cropping to avoid repeated disk I/O could be optimized, 
        # but for simplicity we load and crop here or pre-cache)
        # Optimization: In a real scenario, we'd pre-extract patches or cache images.
        
        img1 = Image.open(patch_info['img1_path']).convert('RGB')
        img2 = Image.open(patch_info['img2_path']).convert('RGB')
        label = Image.open(patch_info['label_path']).convert('L')

        # Crop
        img1_patch = img1.crop((j, i, j + self.patch_size, i + self.patch_size))
        img2_patch = img2.crop((j, i, j + self.patch_size, i + self.patch_size))
        label_patch = label.crop((j, i, j + self.patch_size, i + self.patch_size))

        # Data Augmentation
        if self.transform:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img1_patch = img1_patch.transpose(Image.FLIP_LEFT_RIGHT)
                img2_patch = img2_patch.transpose(Image.FLIP_LEFT_RIGHT)
                label_patch = label_patch.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if np.random.random() > 0.5:
                img1_patch = img1_patch.transpose(Image.FLIP_TOP_BOTTOM)
                img2_patch = img2_patch.transpose(Image.FLIP_TOP_BOTTOM)
                label_patch = label_patch.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (0, 90, 180, 270)
            if np.random.random() > 0.5:
                angle = np.random.choice([0, 90, 180, 270])
                if angle != 0:
                    img1_patch = img1_patch.rotate(angle)
                    img2_patch = img2_patch.rotate(angle)
                    label_patch = label_patch.rotate(angle)

        # Convert to numpy and then to tensor
        img1_patch = np.array(img1_patch).transpose(2, 0, 1) / 255.0
        img2_patch = np.array(img2_patch).transpose(2, 0, 1) / 255.0
        label_patch = np.array(label_patch)
        
        # Handle label values (some might be 0-255 or 0-1)
        # Ensure it's binary: 0 and 1
        label_patch = (label_patch > 0).astype(np.float32)
        label_patch = np.expand_dims(label_patch, axis=0)

        return (
            torch.from_numpy(img1_patch).float(),
            torch.from_numpy(img2_patch).float(),
            torch.from_numpy(label_patch).float()
        )

if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    # Get cities from Images directory
    img_dir = os.path.join(root, "Onera Satellite Change Detection dataset - Images")
    cities = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    
    dataset = OneraDataset(root, cities[:2]) # Test with first 2 cities
    print(f"Total patches: {len(dataset)}")
    if len(dataset) > 0:
        img1, img2, label = dataset[0]
        print(f"Patch shape: {img1.shape}, {label.shape}")
