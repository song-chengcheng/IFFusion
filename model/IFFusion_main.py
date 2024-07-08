import torch
import torch.nn as nn
from model.blocks import CBlock_ln
import numpy as np
class local_vi_encoder(nn.Module):
    def __init__(self, in_dim=3, dim=16):
        super(local_vi_encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        blocks = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05)]
        self.blocks = nn.Sequential(*blocks)
        self.conv2 = nn.Conv2d(dim, 64, 3, padding=1)
    def forward(self, x):
        fea = self.relu(self.conv1(x)) # c 16
        fea = self.blocks(fea)+fea
        out = self.conv2(fea)
        return out

class ir_encoder(nn.Module):
    def __init__(self):
        super(ir_encoder, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(3, 16, 3, 1),
            ConvLayer(16, 32, 3, 1),
            ConvLayer(32, 64, 3, 1)
        )
    def forward(self, x):
        return  self.conv(x)


class img_decoder(nn.Module):
    def __init__(self):
        super(img_decoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.sigmoid(self.conv(x))
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # out = torch.max(out, out*0.2)
        out = self.relu(out)
        return out

class ConvTanh(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTanh, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = torch.tanh(out)/2 + 0.5
        return out

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLayer(128, 32, 3, 1)
        self.conv2 = ConvLayer(32, 32, 3, 1)
        self.conv3 = ConvLayer(64, 32, 3, 1)
        self.conv4 = ConvLayer(96, 32, 3, 1)
    def forward(self, fea_ir, fea_vi):
        fea = self.conv1(torch.cat([fea_ir, fea_vi], dim=1)) # c 32
        out = self.conv2(fea) # 32
        fea = torch.cat([fea, out], dim=1)
        out = self.conv3(fea)
        fea = torch.cat([fea, out], dim=1)
        out = self.conv4(fea)
        return out

class L_net(nn.Module):
    def __init__(self, num=64):
        super(L_net, self).__init__()
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_net(input))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(ConvLayer(32, 24, 3, 1),
                                   ConvLayer(24, 8, 3, 1),
                                   ConvTanh(8, 3, 3, 1))
    def forward(self, fea):
        out = self.conv(fea)
        return out

class Illum_aware(nn.Module):
    def __init__(self, num=64):
        super(Illum_aware, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.illum_aware = nn.Sequential(
            nn.Conv2d(3, num, 2, 2, 0),
            nn.ReLU(),
            nn.Conv2d(num, num, 2, 2, 0),
            nn.ReLU(),
            nn.Conv2d(num, num, 2, 2, 0),
            nn.ReLU()
        )
        self.pre = nn.Conv2d(in_channels=num, out_channels=1, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, input):
        out = self.illum_aware(input)
        out = self.pre(self.avg_pool(out))
        out = torch.sigmoid(out)
        return out

class IFFusion(nn.Module):
    def __init__(self):
        super(IFFusion, self).__init__()
        self.vi_encoder = local_vi_encoder(in_dim=3, dim=16)
        self.ir_encoder = ir_encoder()
        self.vi_decoder = img_decoder()
        self.fuse = DenseBlock()
        self.decoder = Decoder()
        self.ill_aware = Illum_aware(num=64)
        self.L_net = L_net(num=64)
    def forward(self, img_ir, img_vi):
        L = self.L_net(img_vi)
        weight = self.ill_aware(img_vi)
        fea_ir =self.ir_encoder(img_ir)
        fea_vi = self.vi_encoder(img_vi)
        R = self.vi_decoder(fea_vi)
        fea_fuse = self.fuse(fea_ir, fea_vi)
        R_fuse = self.decoder(fea_fuse)
        # out = torch.pow(R, weight) * L
        return R, R_fuse, L, weight


import torchvision
import matplotlib.pyplot as plt
def heatmap(x, name):
    heatmap = x.squeeze()
    plt.imshow(heatmap.cpu().numpy(), cmap='viridis')
    plt.axis('off')
    plt.margins(0)
    path = './results/heatmap/' + name
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

