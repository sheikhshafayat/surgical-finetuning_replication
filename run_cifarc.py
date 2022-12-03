
import pandas as pd
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

from trainer_functions.cifartrainer import evaluate_cifar, train_cifar, build_cifar10
from models.ResNet import ResNetCifar as ResNet
from trainer_functions.cifarCloader import CIFARCDataset



def main():

    # empty dataframe
    df = pd.DataFrame(columns=['corruption', 'accuracy', "lr", "state", "n_subset"])

    curruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]

    n_subset = 3000
    lrs = [1e-3, 1e-4, 1e-5]

    for corruption in curruption_list:
        for lr in lrs:
            cifarc = CIFARCDataset(corruption=corruption)
            ccdataloader = DataLoader(cifarc, batch_size=64, shuffle=True)

            cctrain = torch.utils.data.Subset(cifarc, range(0, n_subset))
            ccvalid = torch.utils.data.Subset(cifarc, range(6000, 8000))
            cctest = torch.utils.data.Subset(cifarc, range(8000, 10000))

            cctrainload = DataLoader(cctrain, batch_size=64, shuffle=True)
            ccvalidload = DataLoader(ccvalid, batch_size=64, shuffle=True)
            cctestload = DataLoader(cctest, batch_size=64, shuffle=True)


            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

            tune_net = build_cifar10(device)

            optimizer = optim.Adam(tune_net.layer3.parameters(), lr=lr, weight_decay=0.0001) # change the optimizer to tune different layers

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # no need to modify this

            checkpoint = torch.load('checkpoints/ckpt.pth')
            tune_net.load_state_dict(checkpoint['net'])


            acc = train_cifar(tune_net, cctrainload, ccvalidload, optimizer, scheduler, device=device, epochs=15)

            # adding results to dataframe 
            dict = {}
            dict['corruption'] = corruption
            dict['accuracy'] = acc
            dict['lr'] = lr
            dict['state'] = "layer3"
            dict['n_subset'] = n_subset
            df_temp = pd.DataFrame(dict, index=[0])
            df = pd.concat([df, df_temp]) 

    # saves the results
    name = "cifarC_" + dict['state'] + str(n_subset) + ".csv"
    df.to_csv(name)       

if __name__ == "__main__":
    main()