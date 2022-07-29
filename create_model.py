import torch.nn as nn
import torch
import torch.nn.functional as F



class iiwpod(nn.Module):
    def __init__(self, ):
        super(iiwpod, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_1 = nn.BatchNorm2d(16, )
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(16, 16, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_2 = nn.BatchNorm2d(16, )
        self.relu_2 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d((2, 2), (2, 2))
        self.conv_3 = nn.Conv2d(16, 32, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_3 = nn.BatchNorm2d(32, )
        self.relu_3 = nn.ReLU()
        self.res_1 = ResBlock(32, 32)
        self.maxpool_2 = nn.MaxPool2d((2, 2), (2, 2))
        self.conv_4 = nn.Conv2d(32, 64, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_4 = nn.BatchNorm2d(64, )
        self.relu_4 = nn.ReLU()
        self.res_2 = ResBlock(64, 64)
        self.res_3 = ResBlock(64, 64)
        self.maxpool_3 = nn.MaxPool2d((2, 2), (2, 2))
        self.conv_5 = nn.Conv2d(64, 64, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_5 = nn.BatchNorm2d(64, )
        self.relu_5 = nn.ReLU()
        self.res_4 = ResBlock(64, 64)
        self.res_5 = ResBlock(64, 64)
        self.head = head()


    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_2(x)
        x = self.maxpool_1(x)
        
        x = self.conv_3(x)
        x = self.batch_3(x)
        x = self.relu_3(x)
        x = self.res_1(x)
        x = self.maxpool_2(x)
        
        x = self.conv_4(x)
        x = self.batch_4(x)
        x = self.relu_4(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.maxpool_3(x)
        
        x = self.conv_5(x)
        x = self.batch_5(x)
        x = self.relu_5(x)
        x = self.res_4(x)
        x = self.res_5(x)
        
        return self.head(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.shortcut = nn.Sequential()
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        shortcut = self.shortcut(x)
        x = self.bn2(self.conv2(x))
        x = x + shortcut
        return nn.ReLU()(x)


class head(nn.Module):
    def __init__(self, ):
        super(head, self).__init__()
        self.conv_1 = nn.Conv2d(64, 32, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_1 = nn.BatchNorm2d(32, )
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(32, 16, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.conv_3 = nn.Conv2d(16, 1, (1, 1), (1, 1), groups = 1, bias=True, padding='same')
        self.sig_1 = nn.Sigmoid()
        
        self.conv_4 = nn.Conv2d(64, 32, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_2 = nn.BatchNorm2d(32)
        self.relu_2 = nn.ReLU()
        self.conv_5 = nn.Conv2d(32, 16, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.conv_6 = nn.Conv2d(16, 6, (1, 1), groups = 1, bias=True, padding='same')

    def forward(self, x):
        xprobs = self.conv_1(x)
        xprobs = self.batch_1(xprobs)
        xprobs = self.relu_1(xprobs)
        xprobs = self.conv_2(xprobs)
        xprobs = self.conv_3(xprobs)
        xprobs = self.sig_1(xprobs)

        xbbox = self.conv_4(x)
        xbbox = self.batch_2(xbbox)
        xbbox = self.relu_2(xbbox)
        xbbox = self.conv_5(xbbox)
        xbbox = self.conv_6(xbbox)

        return torch.cat((xprobs, xbbox), 1)