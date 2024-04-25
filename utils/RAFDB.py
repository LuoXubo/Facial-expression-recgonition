"""
@Description :   RAF-DB dataloader
@Author      :   Xubo Luo 
@Time        :   2024/04/24 08:40:11
"""

import torch
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RAFDB(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join('./data/RAF-DB/DATASET/images/', self.data['image'][idx])
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype('uint8'))
        label = self.data['label'][idx]-1 # 1-7 -> 0-6
        if self.transform:
            img = self.transform(img)

        return img, label

def get_dataloader(data_path, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = RAFDB(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    data_path = './data/RAF-DB/train_labels.csv'
    train_loader = get_dataloader(data_path)
    for x, y in train_loader:
        print(x.shape, y)
        break
