import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from helper_functions import *


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, init='he', num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        if init == 'he':
            print('ResNet He init')
            self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(init='he'):
    return ResNet(BasicBlock, [3, 3, 3], init=init)


class OrgCifarModel(nn.Module):
    # model used in the paper Critial Learning Periods in Deep Networks
    # https://arxiv.org/abs/1711.08856
    def __init__(self):
        super(OrgCifarModel, self).__init__()
        self.features = nn.Sequential(
            # convolution block 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # convolution block 2
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # convolution block 3
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # convolution block 4
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # convolution block 5
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # convolution block 6
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # convolution block 7
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # convolution block 8
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # convolution block 9
            nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def xavier_initialization_weights(model):
    for module_pos, module in model._modules.items():
        if 'Sequential' in str(module):
            for single_layer in module:
                if 'Conv' in str(single_layer):
                    n = np.sqrt(2 / torch.numel(single_layer.weight.data))
                    nn.init.normal_(single_layer.weight.data, mean=0.0, std=n)
                    single_layer.bias.data = single_layer.bias.data * 0.0


if __name__ == "__main__":
    model = resnet20('he')  # OrgCifarModel()
    # A full forward pass
    # [B x C x H x W]
    im = torch.randn(1, 3, 32, 32)
    # model = Cifar_10_Model()
    # model = OrgCifarModel()
    x = model(im)
    print(x)
    save_model(model, '/home/gp/Desktop/model/', 'a.pt')
    # model2 = OrgCifarModel()  # resnet20('he')
    model2 = resnet20('he')
    load_model(model2, '/home/gp/Desktop/model/', 'a.pt')
    x = model(im)
    print(x)
    # print(x.shape)
    del model
    del x
