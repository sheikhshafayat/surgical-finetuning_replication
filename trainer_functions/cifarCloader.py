import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

class CIFARCDataset(Dataset):
    def __init__(self, data_dir='data/CIFAR-10-C', root_dir = "./", corruption="gaussian_blur", transform=None):
        corruption = corruption + ".npy"
        path = os.path.join(root_dir, data_dir, corruption)
        self.alldata = np.load(path)[40000:]
        self.label = np.load(os.path.join(root_dir, data_dir, "labels.npy"))[40000:]
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return self.alldata.shape[0]
    
    def __getitem__(self, idx):
        image = self.alldata[idx]
        label = self.label[idx]
        image = T.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        
        return image, label