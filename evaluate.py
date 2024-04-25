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
# from utils.DataLoader import get_dataloader
from utils.RAFDB import get_dataloader
from models.resnet.resnet_50 import ResNet50
from models.resnet.resnet_101 import ResNet101
from models.resnet.resnet_152 import ResNet152
from models.efficientnet.model import EfficientNet
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='resnet50', help='model name (resnet50, resnet101 or resnet152)')
    parser.add_argument("--test_path", type=str, default='./data/RAF-DB/test_labels.csv', help='test data path')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument("--weight_path", type=str, default='/home/xubo/Codes/course/Facial-expression-recgonition/checkpoints/resnet50/2024_4_22/resnet50_90.pth', help='pretrained model path')
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
    elif method == 'efficientnet0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=7)
    elif method == 'efficientnet4':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=7)
    elif method == 'efficientnet7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=7)
    else:
        raise ValueError('Invalid model name!')
    
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print('%s loaded successfully!' % method)
    print('------------------------------------')

    acc_avg = 0
    cnt = 0
    timecost = 0
    for x_val, y_val in val_loader:

        with torch.no_grad():
            tik = time.time()
            y_pred = model(x_val)
            tok = time.time()
            timecost += (tok - tik)/len(x_val)
            y_pred = torch.argmax(y_pred, dim=1)
            acc = torch.mean((y_pred == y_val).float())
            acc_avg += acc
            cnt += 1
    acc_avg /= cnt
    print('Accuracy of %s on the validation set: %.4f, average timecost: %.4f ' % (method, acc_avg, timecost/cnt))