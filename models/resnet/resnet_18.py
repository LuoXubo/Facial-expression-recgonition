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

ResNet18 = models.resnet18(pretrained=True)
ResNet18.fc = nn.Linear(ResNet18.fc.in_features, num_classes)


if __name__ == '__main__':
    model = ResNet18
    # print(model)
    input = torch.randn(1, 3, 48, 48)
    output = model(input)
    print(output.shape)
    print(output)    