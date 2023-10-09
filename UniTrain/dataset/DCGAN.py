import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class DCGANdataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = self.load_data()

    def load_data(self):
        data = []
        for cls in self.classes:
            class_path = os.path.join(self.data_dir, cls)
            class_idx = self.class_to_idx[cls]
            # Include .jpg, .jpeg, and .png extensions in the glob pattern
            for file_path in glob.glob(os.path.join(class_path, '*.jpg')) + \
                             glob.glob(os.path.join(class_path, '*.jpeg')) + \
                             glob.glob(os.path.join(class_path, '*.png')):
                data.append((file_path, class_idx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target
