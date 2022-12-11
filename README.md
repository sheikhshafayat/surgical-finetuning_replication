## Replication of:
# SURGICAL FINE-TUNING IMPROVES ADAPTATION TO DISTRIBUTION SHIFTS 
We implement the code for the following paper: https://arxiv.org/abs/2210.11466 for CS492-I (Fall 2022) class project in KAIST. There are no available implementations of this paper, so we do it from scratch.

We use pretrained checkpoints for the CIFAR-10 model from the following link: https://github.com/zhangmarvin/memo.

In this repository, there are five experiments in total. 
- CIFAR-C 
- ImageNet-C
- Waterbirds
- Entity-30
- Living-17

Among them, **CIFAR-C** and **Waterbirds** datasets can be found [here](https://kaistackr-my.sharepoint.com/:f:/g/personal/sheikh_shafayat_kaist_ac_kr/EorPF-ZdMlZFm_SYpsE-tWgBghy6kyCEALxzwWYoB2WvbA?e=KghPZj). To create **Entity-30** and **Living-17** datasets, one needs to have the ILSVRC2012 training and validation datasets downloaded from [here] (https://image-net.org/challenges/LSVRC/2012/2012-downloads.php). Detailed guideline on how to make these two datasets can be found [here](https://robustness.readthedocs.io/en/latest/example_usage/breeds_datasets.html). **ImageNet-C** dataset can be found [here](https://zenodo.org/record/2235448).

### To replicate the results of CIFAR-C:
1. Download the dataset from the link provided above and put it in a folder.
2. In trainer_functions/cifarCloader.py change the data_dir to your folder path (modify the root dir too if needed). 
3. Run run_cifar.py. 
4. Inside the file change the optimizer and the dictionary state depending on the layer you are fine-tuning. You can also change n_subset to determine how much data will be used to fine-tune. 
5. The results are now saved in the .csv file.

### To replicate the results of ImageNet-C:
1. Download the dataset from the link provided above and put it in a folder.
2. Change the dataset directory inside ImageNetC_Exp.py and run the file.
3. Similarly to CIFAR-C, change the optimizer and the dictionary state (for more details, see the final report). 
4. The results are now saved in the .csv file.

### To replicate the results of Waterbirds
The procedure is very similar to running experiments on the previous datasets.
1. Download the dataset from the link provided above and put it in a folder.
2. Change csv_dir and root_dir inside run_waterbirds.py according to your dataset location and run the file.
3. Similarly to the previous datasets, change the optimizer and the dictionary state (for more details, see the final report). 
4. The results are now saved in the .csv file.

### To replicate the results of Living-17 and Entity-30
1. Download the ImageNet dataset from the link provided above.
2. Change the dataset directory according to the first step inside the Living17.py or entity30.py files and run the corresponding file.
3. Change the optimizer and the dictionary state as in the previous cases.
4. The results are also saved in the .csv file.





