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

ResNet101 = models.resnet101(pretrained=True)
ResNet101.fc = nn.Linear(ResNet101.fc.in_features, num_classes)


if __name__ == '__main__':
    model = ResNet101
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)
    print(output)    