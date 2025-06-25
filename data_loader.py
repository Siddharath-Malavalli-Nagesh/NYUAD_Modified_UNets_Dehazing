import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
# In data_loader.py

# ... (keep imports and the beginning of the ImageFolder class)

class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=(384, 512), mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        # GT : Ground Truth
        self.GT_paths = root[:-1]+'_GT/' # Assumes GT folder is named like 'train_GT' for 'train'
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root)]
        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        
        # Create the corresponding ground truth path by replacing the directory
        filename = os.path.basename(image_path)
        GT_path = os.path.join(self.GT_paths, filename)

        # Open images as RGB
        image = Image.open(image_path).convert('RGB')
        GT = Image.open(GT_path).convert('RGB')

        # --- Transformations ---
        # For training, apply augmentations
        if self.mode == 'train' and random.random() < self.augmentation_prob:
            # Random horizontal flip
            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)
            
            # Random vertical flip
            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            # You can add more augmentations here like ColorJitter, but apply them only to the input `image`
            transform_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = transform_jitter(image)

        # --- Common Transformations for both train and validation/test ---
        
        # Resize to the desired input size for the model
        transform_resize = T.Resize(self.image_size)
        image = transform_resize(image)
        GT = transform_resize(GT)

        # Convert to tensor (scales pixels to [0, 1])
        transform_to_tensor = T.ToTensor()
        image = transform_to_tensor(image)
        GT = transform_to_tensor(GT)

        # Normalize the input image to [-1, 1]. The GT remains in [0, 1]
        # The model's output will be in [0, 1] thanks to a Sigmoid activation.
        transform_norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        image = transform_norm(image)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


# Modify the get_loader function to accept a tuple for image_size
def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    
    dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'), # Only shuffle for training
                                  num_workers=num_workers)
    return data_loader
