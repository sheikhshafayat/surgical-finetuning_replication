import torch
import torch.nn as nn
from trainer_functions.waterbirdsloader import WaterbirdsDataset
from trainer_functions.waterbirdstrainer import evaluate_waterbirds, train_waterbirds
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wb_train = WaterbirdsDataset(csv_dir = "data/waterbird_complete95_forest2water2/metadata.csv", mode = "train", root_dir="data/waterbird_complete95_forest2water2")
wb_target = WaterbirdsDataset(csv_dir = "data/waterbird_complete95_forest2water2/metadata.csv", mode = "val", root_dir="data/waterbird_complete95_forest2water2")
wbtarget_test = WaterbirdsDataset(csv_dir = "data/waterbird_complete95_forest2water2/metadata.csv", mode = "test", root_dir="data/waterbird_complete95_forest2water2")

# split wbtrain dataset into train and validation
train_size = int(0.8 * len(wb_train))
val_size = len(wb_train) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(wb_train, [train_size, val_size])

# take 400 images from waterbird_target
target_size = 400
target_dataset, target_val = torch.utils.data.random_split(wb_target, [target_size, len(wb_target) - target_size])


print("Train size: ", len(train_dataset))
print("Val size: ", len(val_dataset))
print("Target size: ", len(target_dataset))
print("Target val size: ", len(target_val))
print("Target test size: ", len(wbtarget_test))

# source loader
wb_trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
wb_valloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# target loader
wb_targetloader = DataLoader(target_dataset, batch_size=64, shuffle=True)
wb_target_valloader = DataLoader(target_val, batch_size=64, shuffle=True)

target_testloader = DataLoader(wbtarget_test, batch_size=64, shuffle=True)

# get the pretrained model trained on IMAGENET
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 2) # change the last layer to 2 classes

df = pd.DataFrame(columns=["accuracy", "lr", "state", "n_subset"])
lrs = [0.005, 0.001, 0.0005]


for lr in lrs:
    wb_targetloader = DataLoader(target_dataset, batch_size=64, shuffle=True)
    wb_target_valloader = DataLoader(target_val, batch_size=64, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2)

    # we finetuned the model on this dataset before, so let's load that one
    model.load_state_dict(torch.load("model_states/model_40.pth"))

    # change the optimizer to tune different layers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001) 
    #optimizer = torch.optim.Adam(model.layer1.parameters(), lr=lr, weight_decay=0.0001) 
    #optimizer = torch.optim.Adam(model.layer2.parameters(), lr=lr, weight_decay=0.0001) 
    #optimizer = torch.optim.Adam(model.layer3.parameters(), lr=lr, weight_decay=0.0001) 
    #optimizer = torch.optim.Adam(model.layer4.parameters(), lr=lr, weight_decay=0.0001)
    #optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=0.0001)  


    acc = train_waterbirds(model, wb_targetloader, wb_target_valloader, optimizer, scheduler=None, device=device, epochs=15)

    # adding results to dataframe 
    dict = {}
    dict['accuracy'] = acc
    dict['lr'] = lr
    dict['state'] = "all" # change this to "layer1", "layer2", "layer3", "layer4", "fc", "all"
    dict['n_subset'] = 400
    df_temp = pd.DataFrame(dict, index=[0])
    df = pd.concat([df, df_temp]) 

