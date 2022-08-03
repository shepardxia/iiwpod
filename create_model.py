import torch.nn as nn
import torch
import torch.nn.functional as F



class iiwpod(nn.Module):
    def __init__(self, ):
        super(iiwpod, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_1 = nn.BatchNorm2d(16, )
        self.conv_2 = nn.Conv2d(16, 16, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_2 = nn.BatchNorm2d(16, )
        self.maxpool_1 = nn.MaxPool2d((2, 2), (2, 2))
        self.conv_3 = nn.Conv2d(16, 32, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_3 = nn.BatchNorm2d(32, )
        self.res_1 = ResBlock(32, 32)
        self.maxpool_2 = nn.MaxPool2d((2, 2), (2, 2))
        self.conv_4 = nn.Conv2d(32, 64, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_4 = nn.BatchNorm2d(64, )
        self.res_2 = ResBlock(64, 64)
        self.res_3 = ResBlock(64, 64)
        self.maxpool_3 = nn.MaxPool2d((2, 2), (2, 2))
        self.conv_5 = nn.Conv2d(64, 64, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_5 = nn.BatchNorm2d(64, )
        self.res_4 = ResBlock(64, 64)
        self.res_5 = ResBlock(64, 64)
        self.head = head()


    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = nn.ReLU()(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = nn.ReLU()(x)
        x = self.maxpool_1(x)
        
        x = self.conv_3(x)
        x = self.batch_3(x)
        x = nn.ReLU()(x)
        x = self.res_1(x)
        x = self.maxpool_2(x)
        
        x = self.conv_4(x)
        x = self.batch_4(x)
        x = nn.ReLU()(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.maxpool_3(x)
        
        x = self.conv_5(x)
        x = self.batch_5(x)
        x = nn.ReLU()(x)
        x = self.res_4(x)
        x = self.res_5(x)
        
        return self.head(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        xi = x
        xi = self.conv1(xi)
        xi = self.bn1(xi)
        xi = nn.ReLU()(xi)
        xi = self.conv2(xi)
        xi = self.bn2(xi)

        return nn.ReLU()(xi + x)


class head(nn.Module):
    def __init__(self, ):
        super(head, self).__init__()
        self.conv_0 = nn.Conv2d(64, 64, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_0 = nn.BatchNorm2d(64, )
        self.conv_1 = nn.Conv2d(64, 32, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_1 = nn.BatchNorm2d(32, )
        self.conv_2 = nn.Conv2d(32, 1, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.sig_1 = nn.Sigmoid()
        
        self.conv_3 = nn.Conv2d(64, 64, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_3 = nn.BatchNorm2d(64, )
        self.conv_4 = nn.Conv2d(64, 32, (3, 3), (1, 1), groups = 1, bias=True, padding='same')
        self.batch_4 = nn.BatchNorm2d(32, )
        self.conv_5 = nn.Conv2d(32, 6, (3, 3), (1, 1), groups = 1, bias=True, padding='same')


    def forward(self, x):
        xprobs = self.conv_0(x)
        xprobs = self.batch_0(xprobs)
        xprobs = nn.ReLU()(xprobs)
        xprobs = self.conv_1(xprobs)
        xprobs = self.batch_1(xprobs)
        xprobs = self.conv_2(xprobs)
        xprobs = self.sig_1(xprobs)

        xbbox = self.conv_3(x)
        xbbox = self.batch_3(xbbox)
        xbbox = nn.ReLU()(xbbox)
        xbbox = self.conv_4(xbbox)
        xbbox = self.batch_4(xbbox)
        xbbox = self.conv_5(xbbox)

        return torch.cat((xprobs, xbbox), 1)