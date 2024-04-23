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
from torch.utils.tensorboard import SummaryWriter
from models.resnet.resnet_50 import ResNet50
from models.resnet.resnet_101 import ResNet101
from models.resnet.resnet_152 import ResNet152
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='resnet50', help='model name (resnet50, resnet101 or resnet152)')
    parser.add_argument("--data_path", type=str, default='./data/fer/icml_face_data.csv', help='fer2013 data path')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument("--num_epochs", type=int, default=100, help='number of epochs')
    args = parser.parse_args()

    method = args.method
    data_path = args.data_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    training_date = time.localtime()
    save_path = './checkpoints/%s/%s_%s_%s' % (method, training_date.tm_year, training_date.tm_mon, training_date.tm_mday)
    
    # make log dir
    log_path = './log/%s/%s_%s_%s_%s_%s' % (method, training_date.tm_year, training_date.tm_mon, training_date.tm_mday, training_date.tm_hour, training_date.tm_min)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)


    # load data
    print('Loading data...')
    train_loader = get_dataloader(data_path, batch_size=batch_size)
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
    
    model = model.cuda()
    print('%s loaded successfully!' % method)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print('Start training...')
    for epoch in tqdm.tqdm(range(num_epochs)):
        cnt = 0
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            x_train = x_train.cuda()
            output = model(x_train)
            output = output.cpu()

            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            cnt += len(x_train)
            print('Epoch: {}/{}, iteration: {}/{}, loss:{} '.format(epoch, num_epochs, cnt, len(train_loader)*batch_size, loss.item()))
            
        writer.add_scalar('Loss/train', loss.item(), epoch)

        # save model
        if epoch % 10 == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            model.eval()
            model = model.cpu()
            # acc_sum = 0
            # for x_test, y_test in test_loader:
            #     output = model(x_test)
            #     loss = criterion(output, y_test)
            #     acc = (output.argmax(dim=1) == y_test).float().mean()
            #     acc_sum += acc.item()
            # print('Epoch: {}/{}, test accuracy: {}'.format(epoch, num_epochs, acc_sum / len(test_loader)))
            torch.save(model.state_dict(), save_path + '/%s_%s.pth' % (method, epoch))
            model = model.cuda()
            model.train()
            # writer.add_scalar('Accuracy/test', acc_sum / len(test_loader), epoch)

    print('Training finished!')