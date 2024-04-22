"""
@Description :   evaluation code
@Author      :   Xubo Luo 
@Time        :   2024/04/22 22:17:50
"""

import matplotlib.pyplot as plt
import torch
import warnings
import tqdm
import time
import os
import argparse
from utils.DataLoader import get_dataloader
from models.resnet.resnet_50 import ResNet50
from models.resnet.resnet_101 import ResNet101
from models.resnet.resnet_152 import ResNet152
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='resnet50', help='model name (resnet50, resnet101 or resnet152)')
    parser.add_argument("--test_path", type=str, default='./dataset/test.csv', help='test data path')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument("--weight_path", type=str, default=None, help='pretrained model path')
    args = parser.parse_args()

    method = args.method
    test_path = args.test_path
    batch_size = args.batch_size
    weight_path = args.weight_path


    # load data
    print('Loading data...')
    val_loader = get_dataloader(test_path, batch_size=batch_size)
    print('Data loaded successfully!')

    print('------------------------------------')
    if method == 'resnet50':
        model = ResNet50
    elif method == 'resnet101':
        model = ResNet101
    elif method == 'resnet152':
        model = ResNet152
    else:
        raise ValueError('Invalid model name!')
    
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print('%s loaded successfully!' % method)
   

    acc_avg = 0
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.cuda(), y_val.cuda()
        with torch.no_grad():
            y_pred = model(x_val)
            # loss = torch.nn.CrossEntropyLoss()(y_pred, y_val)
            y_pred = torch.argmax(y_pred, dim=1)
            acc = torch.mean((y_pred == y_val).float())
            acc_avg += acc
    acc_avg /= len(val_loader)
    print('Accuracy of %s on the validation set: %.4f' % (method, acc_avg))