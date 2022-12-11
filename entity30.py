from robustness import datasets
from robustness.tools.breeds_helpers import make_entity30
import torch
import torch.optim as optim
import pandas as pd

from trainer_functions.customImageNettrainer import build_resnet50, train_customImageNet


def main():

    # directory where ImageNet is stored
    data_dir = '/Users/rada/.mxnet/datasets/imagenet'
    # modified hierarchy of ImageNet is used for creating custom ImageNet datasets
    info_dir = './modified'

    n_subset = 1500
    lrs = [0.0005]

    ret = make_entity30(info_dir, split="rand")
    superclasses, subclass_split, label_map = ret
    train_subclasses, test_subclasses = subclass_split

    dataset_source = datasets.CustomImageNet(data_dir, train_subclasses)
    loaders_source = dataset_source.make_loaders(
    workers=10, batch_size=64, shuffle_train=64, shuffle_val=64, subset=n_subset)
    train_loader_source, val_loader_source = loaders_source

    dataset_target = datasets.CustomImageNet(data_dir, test_subclasses)
    loaders_target = dataset_target.make_loaders(
    workers=10, batch_size=64, shuffle_train=True, shuffle_val=True, subset=n_subset)
    train_loader_target, val_loader_target = loaders_target


    df = pd.DataFrame(columns=['accuracy', "lr", "state", "n_subset"])

    for lr in lrs:
        device = torch.device("mps")  # change to cuda for nvidia device
        tune_net = build_resnet50(device)

        # don't modify anything here
        optimizer = optim.Adam(tune_net.fc.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        acc = train_customImageNet(tune_net, train_loader_source,
                            val_loader_source, optimizer, scheduler, device=device, epochs=3)

        # don't modify anything here
        optimizer = optim.Adam(tune_net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        acc = train_customImageNet(tune_net, train_loader_source,
                            val_loader_source, optimizer, scheduler, device=device, epochs=2)

        # choose which layer to tune. Change this according to your experiment
        optimizer = optim.Adam(tune_net.parameters(), lr=lr)
        #optimizer = optim.Adam(tune_net.layer1.parameters(), lr=lr)
        #optimizer = optim.Adam(tune_net.layer2.parameters(), lr=lr)
        #optimizer = optim.Adam(tune_net.layer3.parameters(), lr=lr)
        #optimizer = optim.Adam(tune_net.layer4.parameters(), lr=lr)
        #optimizer = optim.Adam(tune_net.fc.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        acc = train_customImageNet(tune_net, train_loader_target,
                            val_loader_target, optimizer, scheduler, device=device, epochs=15)

        # adding results to dataframe
        dict = {}
        dict['accuracy'] = acc
        dict['lr'] = lr
        dict['state'] = "first layer"  # change when run
        dict['n_subset'] = n_subset
        df_temp = pd.DataFrame(dict, index=[0])
        df = pd.concat([df, df_temp])

    file_name = "Entity30_" + dict['state'] + ".csv"
    df.to_csv(file_name)

if __name__ == "__main__":
    main()
