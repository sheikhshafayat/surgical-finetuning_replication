## Replication of:
# SURGICAL FINE-TUNING IMPROVES ADAPTATION TO DISTRIBUTION SHIFTS 
We replicate the code for this paper: https://arxiv.org/abs/2210.11466 for CS492-I (Fall 2022) class project in KAIST. The code is not avaliable publicly, so we try to do it from scratch.

We use pretrained checkpoint for CIFAR-10 model from here https://github.com/zhangmarvin/memo

In this repository, there are five experiments in total. 
- CIFAR-C 
- ImageNet-C
- Waterbirds
- Entity-30
- Living-17

Among them, **CIFAR-C** and **Waterbirds** dataset can be found in this [link](https://kaistackr-my.sharepoint.com/:f:/g/personal/sheikh_shafayat_kaist_ac_kr/EorPF-ZdMlZFm_SYpsE-tWgBghy6kyCEALxzwWYoB2WvbA?e=KghPZj). Entity-30 and Living-17 requires to have the whole ImageNet dataset downloader.

ImageNet-C dataset can be found [here](https://zenodo.org/record/2235448). Entity-30 and Living-17 are subsets of ImageNet and they require to have the whole ImageNet dataset on disc. Details guidelines for making these two datasets can be found [here](https://robustness.readthedocs.io/en/latest/example_usage/breeds_datasets.html). 

### To replicate the results of CIFAR-C:
Download the dataset from the link and put it in a folder. Then in trainer_functions/cifarCloader.py change the data_dir to your folder (modify the root dir too if needed). After that run run_cifar.py. Inside the file change the optimizer and dict state according to the layer you are tuning. You can also change n_subset to determine how much data would be used to finetune. The results would be saved as a .csv file.

### To replicate the results of ImageNet-C:
Run ImageNetC_Exp.py  and change the dataset directory inside it. Similar to CIFAR-C, change the optimizer and dictionary state (everything is commented out in the paper). The results would be outputteed as csv file.

### To replicate the results of Waterbirds
Very similar to the previous ones: go to run_waterbirds.py and change the csv_dir and root_dir according to your dataset location and then run it. You can change the optimizer and dictionary state. The result would be saved as a csv file. 

### To replicate the results of Living-17 and Entity-30

Run Living17.py and change the dataset directory inside the file. Change the optimizer, dictionary state and save file name accordingly. Entity-30 is also similarr to Living-17, the corresponding file is entity30.py 





