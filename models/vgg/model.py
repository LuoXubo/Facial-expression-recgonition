"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/05/10 16:47:02
"""

 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

vgg16 = models.vgg16(pretrained=True)
#获取VGG16的特征提取层
vgg = vgg16.features
#将vgg16的特征提取层参数冻结，不对其进行更细腻
for param in vgg.parameters():
    param.requires_grad_(False)

#使用VGG16的特征提取层+新的全连接层组成新的网络
class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel,self).__init__()
        #预训练的Vgg16的特征提取层
        self.vgg = vgg
        #添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256,7),

        )

    #定义网络的前向传播
    def forward(self,x):
        x = self.vgg(x)
        x = x.view(x.size(0),-1)
        output = self.classifier(x)
        return output



if __name__ == '__main__':
    model = MyVggModel()
    input = torch.randn(1, 3, 48, 48)
    output = model(input)
    print(output.shape)
    print(output)    

    

