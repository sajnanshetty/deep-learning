import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock


class ResnetCustomCifa10(nn.Module):
    def __init__(self):
        super(ResnetCustomCifa10, self).__init__()

        self.prep_layer = ResnetCustomCifa10.make_convolution_layer(3, 64)

        self.layer1 = ResnetCustomCifa10.make_convolution_layer(64, 128, max_pool=True)
        self.resnet_layer1 = ResnetCustomCifa10.make_resnet_layer(128)

        self.layer2 = ResnetCustomCifa10.make_convolution_layer(128, 256, max_pool=True)

        self.layer3 = ResnetCustomCifa10.make_convolution_layer(256, 512, max_pool=True)
        self.resnet_layer3 = ResnetCustomCifa10.make_resnet_layer(512)

        self.pool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # preparation layer
        x = self.prep_layer(x)
        # layer1
        x = self.layer1(x)
        r1 = self.resnet_layer1(x)
        x = x + r1
        # layer2
        x = self.layer2(x)
        # layer3
        x = self.layer3(x)
        r3 = self.resnet_layer3(x)
        x = x + r3
        # pool
        x = self.pool(x)

        x = x.view(-1, ResnetCustomCifa10.num_flat_features(x))

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    @staticmethod
    def num_flat_features(x):
        sizes = x.size()[1:]
        num_features = 1
        for item in sizes:
            num_features *= item
        return num_features

    @staticmethod
    def make_convolution_layer(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False,
                               max_pool=False):
        conv_layer = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()]
        if max_pool:
            conv_layer.insert(1, nn.MaxPool2d(2, 2))
        return nn.Sequential(*conv_layer)

    @staticmethod
    def make_resnet_layer(channels):
        resnet_layers = BasicBlock(channels, channels)
        return nn.Sequential(resnet_layers)
