import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd

from trainer_functions.imagenetc import ImageNetC
from trainer_functions.imagenetctrainer import build_resnet50, train_imagenetc

def main():
    # corruptions used to establish the best learning rate
    # corruption_list = ["frost", "gaussian_noise", "glass_blur", "impulse_noise", "snow"]
    # lrs = [1e-3, 1e-4, 1e-5]

    # all other corruptions; we run experiments with the best learning rate
    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "jpeg_compression", "motion_blur", "pixelate", "shot_noise",  "zoom_blur"]
    lrs = [1e-3]

    n_subset = 2000 # number of images to fine-tune on: try 1000, 2000, or 3000

    df = pd.DataFrame(columns=['corruption', 'accuracy', "lr", "state", "n_subset"])

    for corruption in corruption_list:
        
        # put the root of ImageNet-C dataset here
        dataset = ImageNetC(root='/Users/rada/Desktop/KAIST/Deep Learning/Datasets/ImageNetC',
                            corruption=corruption, severity=5,
                            transform=T.Compose([T.ToTensor()])
                            )

        indices = torch.randperm(len(dataset))
        for lr in lrs:

            ictrain = torch.utils.data.Subset(dataset, indices[:n_subset]) # P_target 
            icvalid = torch.utils.data.Subset(dataset, indices[n_subset:n_subset+2000]) # Validation data
            ictest = torch.utils.data.Subset(dataset, indices[n_subset+2000:n_subset+4000]) # Test data, which we don't use here

            ictrainload = DataLoader(ictrain, batch_size=64, shuffle=True)
            icvalidload = DataLoader(icvalid, batch_size=64, shuffle=True)
            ictestload = DataLoader(ictest, batch_size=64, shuffle=True)

            
            device = torch.device("mps") # all experiments were done on a Mac, setting it to "mps"
            tune_net = build_resnet50(device) # build a ResNet-50 model default pretrained on ImageNet

            # Choose which layer to tune. Total 6 options
            
            optimizer = optim.Adam(tune_net.layer1.parameters(), lr=lr, weight_decay=0.0001)
            #optimizer = optim.Adam(tune_net.layer2.parameters(), lr=lr, weight_decay=0.0001)
            #optimizer = optim.Adam(tune_net.layer3.parameters(), lr=lr, weight_decay=0.0001)
            #optimizer = optim.Adam(tune_net.layer4.parameters(), lr=lr, weight_decay=0.0001)
            #optimizer = optim.Adam(tune_net.fc.parameters(), lr=lr, weight_decay=0.0001)
            #optimizer = optim.Adam(tune_net.parameters(), lr=lr, weight_decay=0.0001)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

            acc = train_imagenetc(tune_net, ictrainload, icvalidload, optimizer, scheduler, device=device, epochs=10)

            # adding results to dataframe
            dict = {}
            dict['corruption'] = corruption
            dict['accuracy'] = acc
            dict['lr'] = lr
            dict['state'] = "first layer" # change this when run for different layers
            dict['n_subset'] = n_subset
            df_temp = pd.DataFrame(dict, index=[0])
            df = pd.concat([df, df_temp])

    file_name = "ImageNetC_" + dict['state'] + ".csv"
    df.to_csv(file_name)

if __name__ == "__main__":
    main()
