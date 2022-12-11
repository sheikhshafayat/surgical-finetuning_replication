from collections import OrderedDict
from robustness import datasets
from robustness.tools.breeds_helpers import make_living17
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
import os
import torchvision
import pandas as pd
# for plotting

from trainer_functions.customImageNettrainer import build_resnet50, evaluate_customImageNet, train_customImageNet

data_dir = '/Users/rada/.mxnet/datasets/imagenet'
info_dir = '/Users/rada/Documents/GitHub/BREEDS-Benchmarks/imagenet_class_hierarchy/modified'

n_subset = 850
lr = 0.0005


# empty dataframe for storing results
df = pd.DataFrame(columns=['accuracy', "lr", "state", "n_subset"])


# making the dataset and dataloaders
ret = make_living17(info_dir, split="rand")
superclasses, subclass_split, label_map = ret
train_subclasses, test_subclasses = subclass_split

dataset_source = datasets.CustomImageNet(data_dir, train_subclasses)
loaders_source = dataset_source.make_loaders(
    workers=10, batch_size=64, shuffle_train=64, shuffle_val=64, subset=850)
train_loader_source, val_loader_source = loaders_source

dataset_target = datasets.CustomImageNet(data_dir, test_subclasses)
loaders_target = dataset_target.make_loaders(
    workers=10, batch_size=64, shuffle_train=True, shuffle_val=True, subset=850)
train_loader_target, val_loader_target = loaders_target

device = torch.device("mps") # change to cuda or cpu if it is not a mac
tune_net = build_resnet50(device)

# don't change anything here (not related to finetuning) 
optimizer = optim.Adam(tune_net.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
acc = train_customImageNet(tune_net, train_loader_source,
                           val_loader_source, optimizer, scheduler, device=device, epochs=3)

# don't change anything here (don't change anything here), not related to finetuning
optimizer = optim.Adam(tune_net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
acc = train_customImageNet(tune_net, train_loader_source,
                           val_loader_source, optimizer, scheduler, device=device, epochs=2)


# change the layer you want to finetune here

optimizer = optim.Adam(tune_net.parameters(), lr=lr)
#optimizer = optim.Adam(tune_net.layer1.parameters(), lr=lr)
#optimizer = optim.Adam(tune_net.layer2.parameters(), lr=lr)
#optimizer = optim.Adam(tune_net.layer3.parameters(), lr=lr)
#optimizer = optim.Adam(tune_net.layer4.parameters(), lr=lr)
#optimizer = optim.Adam(tune_net.fc.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
acc = train_customImageNet(tune_net, train_loader_target,
                           val_loader_target, optimizer, scheduler, device=device, epochs=15)

model_scripted = torch.jit.script(tune_net) # Export to TorchScript
model_scripted.save('Living17_layer4.pt')

# adding results to dataframe
dict = {}
dict['accuracy'] = acc
dict['lr'] = lr
dict['state'] = "fourth layer"  # change when run
dict['n_subset'] = n_subset
df_temp = pd.DataFrame(dict, index=[0])
df = pd.concat([df, df_temp])

# change when run
df.to_csv("Living17_layer4.csv")