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


from trainer_functions.cifartrainer import evaluate_cifar, train_cifar, build_cifar10, build_cifar10noise
from models.ResNet import ResNetCifar as ResNet


NUM_TRAIN = 45000


transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We download CIFAR-C dataset
cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
# We don't use the training dataset

# 3000 validation images will be used for fine tuning and 2000 for validation
cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 48000)))
loader_val_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(48000, 50000)))

# test set
cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

lr = 1e-3
device = torch.device('cpu') # you can use 'cuda' if you have GPU
tune_net = build_cifar10noise(device)
checkpoint = torch.load('checkpoints/ckpt.pth', map_location=device) # preload checkpoints from MEMO paper


# The following script is only for one kind of input noise. 
# To change the noise input, go to models/ResNetNoise.py and change in the forward function

for i in range(5):
    
    tune_net.load_state_dict(checkpoint['net'])
    
    optims = [optim.Adam(tune_net.parameters(), lr=lr, weight_decay=0.0001), optim.Adam(tune_net.layer1.parameters(), lr=lr, weight_decay=0.0001), optim.Adam(tune_net.layer2.parameters(), lr=lr, weight_decay=0.0001),
    optim.Adam(tune_net.layer3.parameters(), lr=lr, weight_decay=0.0001), optim.Adam(tune_net.fc.parameters(), lr=lr, weight_decay=0.0001)]
    optimizer = optims[i]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    acc = train_cifar(tune_net, loader_val, loader_val_val, optimizer, scheduler, device=device, epochs=15)


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
    dict['noise'] = "layer3"
    df_temp = pd.DataFrame(dict, index=[0])
    df = pd.concat([df, df_temp])
    names = ["all", "layer1", "layer2", "layer3", "fc"]
    file_name = "trialcifar10noise_" + dict['noise'] + "_" +  names[i] + ".csv"

    df.to_csv(file_name)