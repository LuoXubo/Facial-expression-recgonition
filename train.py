"""
@Description :   training code
@Author      :   Xubo Luo 
@Time        :   2024/04/22 19:22:54
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
    parser.add_argument("--train_path", type=str, default='./dataset/train.csv', help='train data path')
    parser.add_argument("--test_path", type=str, default='./dataset/test.csv', help='test data path')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument("--num_epochs", type=int, default=100, help='number of epochs')
    args = parser.parse_args()

    method = args.method
    train_path = args.train_path
    test_path = args.test_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    training_date = time.localtime()
    save_path = './checkpoints/%s/%s_%s_%s' % (method, training_date.tm_year, training_date.tm_mon, training_date.tm_mday)


    # load data
    print('Loading data...')
    train_loader = get_dataloader(train_path, batch_size=batch_size)
    test_loader = get_dataloader(test_path, batch_size=batch_size)
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
    
    print('%s loaded successfully!' % method)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print('Start training...')
    for epoch in tqdm.tqdm(range(num_epochs)):
        cnt = 0
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            output = model(x_train)

            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            cnt += len(x_train)
            print('Epoch: {}/{}, iteration: {}/{}, loss:{} '.format(epoch, num_epochs, cnt, len(train_loader)*batch_size, loss.item()))

        # save model
        if epoch % 10 == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            torch.save(model.state_dict(), save_path + '/%s_%s.pth' % (method, epoch))
            print('Model saved successfully!')

    print('Training finished!')