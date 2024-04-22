"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/04/22 18:58:47
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

num_classes = 7

ResNet50 = models.resnet50(pretrained=True)
ResNet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
ResNet50.fc = nn.Linear(ResNet50.fc.in_features, num_classes)


if __name__ == '__main__':
    model = ResNet50
    # print(model)
    input = torch.randn(1, 3, 48, 48)
    output = model(input)
    print(output.shape)
    print(output)    