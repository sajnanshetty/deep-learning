import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

#function which handles whether to apply BN or GBN
def norm2d(output_channels, batch_type="BN"):
    if batch_type == "GBN":
        num_splits = 8
        return GhostBatchNorm(output_channels,num_splits)
    else:
        return nn.BatchNorm2d(output_channels)

class NetMNIST(nn.Module):
    def __init__(self, batch_type="BN"):
        super(NetMNIST, self).__init__()
        p = 0.07
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(12),
            norm2d(12, batch_type),
            nn.ReLU(),
            # nn.Dropout2d(p)
        ) # output_size = 26, RF=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(12),
            norm2d(12, batch_type),
            nn.ReLU(),
            # nn.Dropout2d(p)
        ) # output_size = 24, RF=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(12),
            norm2d(12, batch_type),
            nn.ReLU(),
            # nn.Dropout2d(p)
        ) # output_size = 22,  RF=7
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
            #nn.BatchNorm2d(12),
            norm2d(12, batch_type),
            nn.ReLU(),
            # nn.Dropout2d(p)
        ) # output_size = 22,  RF=7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11, RF=8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(16),
            norm2d(16, batch_type),
            nn.ReLU(),
            # nn.Dropout2d(p)
        ) # output_size = 9, RF=10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            #nn.BatchNorm2d(16),
            norm2d(16, batch_type),
            nn.ReLU(),
            # nn.Dropout2d(p)
        ) # output_size = 7, RF=14
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            #nn.BatchNorm2d(16),
            norm2d(16, batch_type),
            nn.ReLU(),
            # nn.Dropout2d(p)
        ) # output_size = 7, RF=18
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        ) # output_size = 1, RF=30

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, RF=30

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        #x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class CIFARNet1(nn.Module):
    """This model applies depthwise and dialated convolution"""
    def __init__(self, batch_type="BN", droupout=0.1):
        super(CIFARNet1, self).__init__()
        # Input Block

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),  # outputsize=32 RF=3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(droupout),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),  # outputsize=32 RF=5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(droupout)
        )  # output_size = 32
        self.pool1 = nn.Sequential(nn.MaxPool2d(2, 2),  # outputsize=16 RF=6
                                   nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), bias=False),  # outputsize=16 RF=6
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Dropout2d(droupout),
                                   )  # output=16 RF=6
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),  # output=16 RF=10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(droupout),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                      padding=1, bias=False,
                      groups=64),  # Apply Depthwise convolution #output=16 RF=14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(droupout),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1),  # Apply pointwise convolution on depthwise #output=20 RF=14
                      padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(droupout),

        )  # output_size = 18
        self.pool2 = nn.Sequential(nn.MaxPool2d(2, 2),  # output 10 RF=16
                                   nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Dropout2d(droupout), )  # output 12 RF=16
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False, dilation=2), #Apply dilated convolution  # output 12 RF=24
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(droupout),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, bias=False),  # output 14 RF=32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(droupout)
        )
        self.pool3 = nn.Sequential(nn.MaxPool2d(2, 2),  # output 7 RF=36
                                   nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Dropout2d(droupout))  # output 7 RF=9

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, groups=32),  # output=9 RF=52  # Apply depthwise convolutuin grouping
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(droupout),
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(1, 1), bias=False),  # output=9  RF=52 #Apply pointwise convolution on depthwise
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(droupout),

            nn.AvgPool2d(kernel_size=7), # output_size = 1 RF=100
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 1 RF=100

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.convblock4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)