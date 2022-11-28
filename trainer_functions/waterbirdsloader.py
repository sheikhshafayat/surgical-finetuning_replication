# waterbirds dataset loader

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

class WaterbirdsDataset(Dataset):
    def __init__(self, csv_dir, root_dir = "./", mode="train", transform=None):
        metadata = pd.read_csv(csv_dir)
        if mode == "test":
            self.metadata = metadata[metadata['split'] == 2]
        elif mode == "val":
            self.metadata = metadata[metadata['split'] == 1]
        else:
            self.metadata = metadata[metadata['split'] == 0]
        
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx, 1])
        image = PIL.Image.open(img_name)
        label = self.metadata.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        image = transforms.ToTensor()(image)
        image = transforms.CenterCrop(224)(image)
        return image, label