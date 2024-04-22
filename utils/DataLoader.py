"""
@Description :   Load the data
@Author      :   Xubo Luo 
@Time        :   2024/04/22 20:39:35
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FER2013(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data['pixels'][idx]
        img = np.array(img.split(' '))
        img = img.reshape(48, 48)
        img = Image.fromarray(img.astype('uint8'))
        label = self.data['emotion'][idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
def get_dataloader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = FER2013(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader




if __name__ == '__main__':
    data_path = '../dataset/icml_face_data.csv'
    dataloader  = get_dataloader(data_path, 32)
    for img, label in dataloader:
        print(img.shape)
        print(label)
        break