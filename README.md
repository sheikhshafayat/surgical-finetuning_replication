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

Among them, CIFAR-C and Waterbirds dataset can be found in this [link](https://kaistackr-my.sharepoint.com/:f:/g/personal/sheikh_shafayat_kaist_ac_kr/EorPF-ZdMlZFm_SYpsE-tWgBghy6kyCEALxzwWYoB2WvbA?e=KghPZj). Entity-30 and Living-17 requires to have the whole ImageNet dataset downloader
