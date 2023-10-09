import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image and mask based on index
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        print(image_path)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img = cv2.imread(image_path)

        if self.transform is not None:
            image = self.transform(image)
            
        mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        mask = mask_transform(mask)
        mask = mask.to(torch.long)
        
        # You may need to further preprocess the mask if required
        # Example: Convert mask to tensor and perform class mapping
        return image, mask
