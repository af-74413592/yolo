import torch
from torch import nn
from torch.nn import functional

cls_num = 20
channel_dim = 3*(cls_num+5)

class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,paddding,bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,paddding,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.sub_module(x)

class ResidualLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResidualLayer, self).__init__()
        self.submodule = nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),
        )
    def forward(self,x):
        return self.submodule(x)+x

class ConvlutionalSetLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvlutionalSetLayer, self).__init__()
        # out_channels = in_channels//2
        self.submodule = nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )
    def forward(self,x):
        return self.submodule(x)

class DownSamplingLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownSamplingLayer, self).__init__()
        self.submodule = nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,3,2,1)
        )
    def forward(self,x):
        return self.submodule(x)

class UpSamplingLayer(nn.Module):
    def __init__(self):
        super(UpSamplingLayer, self).__init__()

    def forward(self,x):
        return functional.interpolate(x,scale_factor=2,mode='nearest')

class Yolo_V3_Net(nn.Module):
    def __init__(self):
        super(Yolo_V3_Net, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3,32,3,1,1),
            DownSamplingLayer(32,64),
            ResidualLayer(64,32),
            DownSamplingLayer(64,128),
            ResidualLayer(128,64),
            ResidualLayer(128,64),
            DownSamplingLayer(128,256),
            ResidualLayer(256,128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128)
        )
        self.trunk_26 = nn.Sequential(
            DownSamplingLayer(256,512),
            ResidualLayer(512,256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256)
        )
        self.trunk_13 = nn.Sequential(
            DownSamplingLayer(512,1024),
            ResidualLayer(1024,512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
        )
        self.convset_13 = nn.Sequential(
            ConvlutionalSetLayer(1024,512)
        )
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,channel_dim,1,1,0)
        )
        self.up_13_to_26 = nn.Sequential(
            ConvolutionalLayer(512,256,3,1,1),
            UpSamplingLayer()
        )
        self.convset_26 = nn.Sequential(
            ConvlutionalSetLayer(768,256)
        )
        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, channel_dim, 1, 1, 0)
        )
        self.up_26_to_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 3, 1, 1),
            UpSamplingLayer()
        )
        self.convset_52 = nn.Sequential(
            ConvlutionalSetLayer(384,128)
        )
        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128,256,3,1,1),
            nn.Conv2d(256,channel_dim,1,1,0)
        )
    def forward(self,x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_13_out = self.convset_13(h_13)
        detection_13_out = self.detection_13(convset_13_out)
        up_13_to_26_out = self.up_13_to_26(convset_13_out)
        cat_13_to_26_out = torch.cat((up_13_to_26_out,h_26),dim=1)
        convset_26_out = self.convset_26(cat_13_to_26_out)
        detection_26_out = self.detection_26(convset_26_out)
        up_26_to_52_out = self.up_26_to_52(convset_26_out)
        cat_26_to_52_out = torch.cat((up_26_to_52_out,h_52),dim=1)
        convset_52_out = self.convset_52(cat_26_to_52_out)
        detection_52_out = self.detection_52(convset_52_out)

        return detection_13_out,detection_26_out,detection_52_out

if __name__ == '__main__':
    net = Yolo_V3_Net()
    x = torch.randn(1,3,416,416)
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)