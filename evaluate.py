"""
@Description :   evaluation code
@Author      :   Xubo Luo 
@Time        :   2024/04/22 22:17:50
"""

import matplotlib.pyplot as plt
import torch
import warnings
from tqdm import tqdm
import time
import os
import argparse
# from utils.DataLoader import get_dataloader
from utils.RAFDB import get_dataloader
from models.resnet.resnet_18 import ResNet18
from models.resnet.resnet_50 import ResNet50
from models.resnet.resnet_101 import ResNet101
from models.resnet.resnet_152 import ResNet152
from models.efficientnet.model import EfficientNet
from models.vit.model import ViT
warnings.filterwarnings("ignore")

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


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
    if method == 'resnet18':
        model = ResNet18
    elif method == 'resnet50':
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
    elif method == 'vit':
        model = ViT(
        image_size = 100,
        patch_size = 10,
        num_classes = 7,
        dim = 1024,
        depth = 3,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    else:
        raise ValueError('Invalid model name!')
    
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print('%s loaded successfully!' % method)
    print('------------------------------------')

    acc_avg = 0
    cnt = 0
    timecost = 0
    num_emotions = [0]*7
    num_correct = [0]*7
    with tqdm(total=len(val_loader)) as pbar:
        for x_val, y_val in val_loader:
            with torch.no_grad():
                tik = time.time()
                y_pred = model(x_val)
                tok = time.time()
                timecost += (tok - tik)/len(x_val)
                y_pred = torch.argmax(y_pred, dim=1)
                acc = torch.mean((y_pred == y_val).float())
                acc_avg += acc
                for idx in range(len(y_val)):
                    num_emotions[y_val[idx]] += 1
                    if y_pred[idx] == y_val[idx]:
                        num_correct[y_val[idx]] += 1
                cnt += 1
            pbar.update(1)

    acc_avg /= cnt
    with open('./results.txt', 'a') as f:
        f.write('Accuracy of %s on the validation set: %.4f, average timecost: %.4f \n' % (method, acc_avg, timecost/cnt))
        print('Accuracy of %s on the validation set: %.4f, average timecost: %.4f ' % (method, acc_avg, timecost/cnt))
        print('------------------------------------')
        for i in range(7):
            print('Emotion: %s, Total: %d, Correct: %d, Accuracy: %.4f' % (emotions[i], num_emotions[i], num_correct[i], num_correct[i]/max(num_emotions[i],1)))
            f.write('Emotion: %s, Total: %d, Correct: %d, Accuracy: %.4f \n' % (emotions[i], num_emotions[i], num_correct[i], num_correct[i]/max(num_emotions[i],1)))
        
    f.close()