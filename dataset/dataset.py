import torch
import torch.nn as nn
import os
import glob
from PIL import Image
import time
import cv2

class Retina_dataset(object):
    def __init__(self, input_dir,transform, input_format='RGB', output_format='RGB'):
        
        self.input_imgs=[]
        self.input_format=input_format
        self.output_format=output_format
        self.transform=transform

        if input_dir is not None:

            for ext in ['jpeg', 'jpg', 'png', 'JPEG', 'PNG', 'JPG']:
                self.input_imgs+=sorted(glob.glob('{}/*.{}'.format(input_dir, ext)))

        # self.transform_rot=[0, 90, 180, 270]

    def __getitem__(self,idx):
        img=Image.open(self.input_imgs[idx])


        start_time=time.time()
        if self.transform:
            img=self.transform(img)

        total_time=time.time()-start_time
        return img, total_time


    def __len__(self):
        return print(len(self.input_imgs))



class Retina_dataset_albumentation(object):
    def __init__(self, input_dir,transform, input_format='RGB', output_format='RGB'):
        
        self.input_imgs=[]
        self.input_format=input_format
        self.output_format=output_format
        self.transform=transform

        if input_dir is not None:

            for ext in ['jpeg', 'jpg', 'png', 'JPEG', 'PNG', 'JPG']:
                self.input_imgs+=sorted(glob.glob('{}/*.{}'.format(input_dir, ext)))

        # self.transform_rot=[0, 90, 180, 270]

    def __getitem__(self,idx):
        img=cv2.imread(self.input_imgs[idx])
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img=Image.open(self.input_imgs[idx])


        start_time=time.time()
        if self.transform:
            augmented=self.transform(image=img)
            img=augmented['image']

            total_time=time.time()-start_time
        return img, total_time


    def __len__(self):
        return print(len(self.input_imgs))