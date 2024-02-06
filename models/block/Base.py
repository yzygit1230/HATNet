import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BasicConv', 'Conv1x1', 'Conv3x3', 'Conv7x7', 'MaxPool2x2', 'MaxUnPool2x2', 'ConvTransposed3x3']


def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d

def get_act_layer():
    # TODO: select appropriate activation layer
    return nn.ReLU

def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)

def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)

class BasicConv(nn.Module):
    def __init__(
        self, in_ch, out_ch, 
        kernel_size, pad_mode='Zero', 
        bias='auto', norm=False, act=False, 
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize()+'Pad2d')(kernel_size//2))
        seq.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=1, padding=0,
                bias=(False if norm else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

class Conv1x1(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 1, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class Conv7x7(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 7, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class MaxPool2x2(nn.MaxPool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)

class MaxUnPool2x2(nn.MaxUnpool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)

class ConvTransposed3x3(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        bias='auto', norm=False, act=False, 
        **kwargs
    ):
        super().__init__()
        seq = []
        seq.append(
            nn.ConvTranspose2d(
                in_ch, out_ch, 3,
                stride=2, padding=1,
                bias=(False if norm else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return F.sigmoid(x)

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class Conv1Relu(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super(Conv1Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = None
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.block(x) + residual
        x = self.relu(x)
        return x

class ChannelChecker(nn.Module):
    def __init__(self, backbone, inplanes, input_size):
        super(ChannelChecker, self).__init__()
        input_sample = torch.randn(1, 3, input_size, input_size)
        f1, f2, f3, f4 = backbone(input_sample)

        channels1 = f1.size(1)
        channels2 = f2.size(1)
        channels3 = f3.size(1)
        channels4 = f4.size(1)

        self.conv1 = Conv1Relu(channels1, inplanes) if (channels1 != inplanes) else None
        self.conv2 = Conv1Relu(channels2, inplanes*2) if (channels2 != inplanes*2) else None
        self.conv3 = Conv1Relu(channels3, inplanes*4) if (channels3 != inplanes*4) else None
        self.conv4 = Conv1Relu(channels4, inplanes*8) if (channels4 != inplanes*8) else None

    def forward(self, f1, f2, f3, f4):
        f1 = self.conv1(f1) if (self.conv1 is not None) else f1
        f2 = self.conv2(f2) if (self.conv2 is not None) else f2
        f3 = self.conv3(f3) if (self.conv3 is not None) else f3
        f4 = self.conv4(f4) if (self.conv4 is not None) else f4

        return f1, f2, f3, f4
