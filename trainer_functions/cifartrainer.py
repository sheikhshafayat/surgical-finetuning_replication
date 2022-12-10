# part of this code is from https://github.com/zhangmarvin/memo

from models.ResNet import ResNetCifar as ResNet
from models.ResNetNoise import ResNetNoise as ResNetNoise
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T



def build_cifar10(device):
    print('Building model...')
    def gn_helper(planes):
        return nn.GroupNorm(8, planes)
    net = ResNet(26, 1, channels=3, classes=10, norm_layer=gn_helper).to(device)
    return net

def build_cifar10noise(device):
    print('Building model...')
    def gn_helper(planes):
        return nn.GroupNorm(8, planes)
    net = ResNetNoise(26, 1, channels=3, classes=10, norm_layer=gn_helper).to(device)
    return net



def evaluate_cifar(loader, model, device):
    
  num_correct = 0
  num_samples = 0
  dtype = torch.float
  ltype = torch.long
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
    for x, y in loader:

      x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
      y = y.to(device=device, dtype=ltype)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
  return acc

def train_cifar(model, train_loader, val_loader, optimizer, scheduler, device, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Accuracy.
    """
    
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    acc = 0
    dtype = torch.float
    ltype = torch.long
    for e in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        scheduler.step()


        print('Epoch %d, loss = %.4f, lr %.8f' % (e, loss.item(), scheduler.get_last_lr()[0]))
        acc = evaluate_cifar(val_loader, model, device)

        print()
    return acc #, losses, accuracies 