from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
import os
import torchvision.datasets as dset
import torchvision.transforms as T
import pandas as pd
# for plotting
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from trainer_functions.cifartrainer import evaluate_cifar, train_cifar, build_cifar10, build_cifar10noise
from models.ResNet import ResNetCifar as ResNet


NUM_TRAIN = 45000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 48000)))
loader_val_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(48000, 50000)))

cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)


lr = 1e-3
device = torch.device('cpu')
tune_net = build_cifar10noise(device)
checkpoint = torch.load('checkpoints/ckpt.pth', map_location=device)


for i in range(4):
    
    tune_net.load_state_dict(checkpoint['net'])
    
    optims = [optim.Adam(tune_net.parameters(), lr=lr, weight_decay=0.0001), optim.Adam(tune_net.layer1.parameters(), lr=lr, weight_decay=0.0001), optim.Adam(tune_net.layer2.parameters(), lr=lr, weight_decay=0.0001),
    optim.Adam(tune_net.layer3.parameters(), lr=lr, weight_decay=0.0001), optim.Adam(tune_net.fc.parameters(), lr=lr, weight_decay=0.0001)]
    optimizer = optims[i]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    acc = train_cifar(tune_net, loader_val, loader_val_val, optimizer, scheduler, device=device, epochs=15)

    print("##########")
    print("##########")
    print("Validation Accuracy: ", acc)
    test_acc = evaluate_cifar(loader_test, tune_net, device)
    print("Test Accuracy: ", test_acc)
    print()
    # adding results to dataframe
    df = pd.DataFrame(columns=['val_accuracy', "lr", "state", "noise", "test_accuracy"]) 
    dict = {}
    dict['val_accuracy'] = acc
    dict['test_accuracy'] = test_acc
    dict['lr'] = lr
    dict['state'] = "all"
    dict['noise'] = "fc"
    df_temp = pd.DataFrame(dict, index=[0])
    df = pd.concat([df, df_temp])
    names = ["all", "layer1", "layer2", "layer3", "fc"]
    file_name = "trialcifar10noise_" + dict['noise'] + "_" +  names[i] + ".csv"

    df.to_csv(file_name)