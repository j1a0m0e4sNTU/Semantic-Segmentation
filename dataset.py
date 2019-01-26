import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import os
from matplotlib import pyplot as plt
from util import *

class Dataset_Seg(Dataset):
    def __init__(self, root, train = True):
        super(Dataset_Seg, self).__init__()
        data_dir = None
        if train == True:
            data_dir = os.path.join(root,'train')
        else:
            data_dir = os.path.join(root, 'validation')
        
        self.img_list = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('sat.jpg')]
        self.msk_list = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('mask.png')]

        self.img_list.sort()
        self.msk_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img   = read_sat_image(self.img_list[idx])
        label = read_mask_to_label(self.msk_list[idx])
        return img, label

def unit_test():
    data_train = Dataset_Seg('../data-segmentation', train=True)
    data_valid = Dataset_Seg('../data-segmentation', train= False)
    loader = DataLoader(data_train, batch_size= 8)
    i = 0
    for imgs, msks in loader:
        if i == 2:break
        i = i+1
        print(imgs.size())
        print(msks.size()) 

if __name__ == '__main__':
    unit_test()