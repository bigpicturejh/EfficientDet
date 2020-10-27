import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from dataset.dataset import Retina_dataset, Retina_dataset_albumentation
from torchvision import transforms
import albumentations
from albumentations.pytorch import ToTensorV2

input_dir='/ssd_data/720p_CDJ'
data_transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

data_transfrom_albumentation=albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.RandomCrop(224, 224),
    albumentations.HorizontalFlip(),
    ToTensorV2()
    # albumentations.pytorch.transforms.ToTensor()
    
])


# dataset=Retina_dataset(input_dir, data_transform)
dataset_albumentation=Retina_dataset_albumentation(input_dir, data_transfrom_albumentation)

total_time=0
for i in range(100):
    # trans_img, time=dataset[0]
    trans_img, time=dataset_albumentation[0]
    total_time+=time

print(f'Time consumption is {total_time}')

