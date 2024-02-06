import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import ASPP
from models.BSDE import BSDE

class GFA(nn.Module):
    def __init__(self,  inplane):
        super().__init__()

        self.conv_ir = nn.Conv2d(inplane, inplane * 2, 1, 1, bias=False)
        self.conv_vi = nn.Conv2d(inplane, inplane * 2, 1, 1, bias=False)
        self.conv_fusion = nn.Conv2d(inplane * 4, inplane * 2, 1, 1)
        self.spatial_select = nn.Conv2d(inplane * 6, 2, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ir, x_vi):
        ir = self.conv_ir(x_ir)
        vi = self.conv_vi(x_vi)
        fuse = torch.cat([ir, vi], 1)
        fuse = self.conv_fusion(fuse)
        
        vi_g = vi.mean([2, 3], keepdim=True).expand(-1, -1, vi.shape[2], vi.shape[3])
        ir_g = ir.mean([2, 3], keepdim=True).expand(-1, -1, ir.shape[2], ir.shape[3])
        fuse = torch.cat([fuse, ir_g, vi_g], 1)

        # prob = self.spatial_select(fuse).softmax(1)  # [B, 2, H, W]
        prob = self.sigmoid(self.spatial_select(fuse))
        prob_ir, prob_vi = prob[:, :1], prob[:, 1:]  # 2x [B, 1, H, W]
        x = x_ir * prob_ir + x_vi * prob_vi
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )

class CFFI(nn.Module):
    def __init__(self, inplanes, num_class):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.down3=conv3x3_bn_relu( inplanes * 8,inplanes * 4)
        self.down2=conv3x3_bn_relu( inplanes * 4,inplanes * 2)
        self.down1=conv3x3_bn_relu( inplanes * 2,inplanes * 1)

        self.bsde4 = BSDE(in_planes=inplanes * 8, out_planes=inplanes * 8)
        self.bsde3 = BSDE(in_planes=inplanes * 4, out_planes=inplanes * 4)
        self.bsde2 = BSDE(in_planes=inplanes * 2, out_planes=inplanes * 2)
        self.bsde1 = BSDE(in_planes=inplanes * 1, out_planes=inplanes * 1)

        self.gfa41 = GFA(inplanes)
        self.gfa31 = GFA(inplanes)
        self.gfa21 = GFA(inplanes)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.expand_field = ASPP(inplanes * 8)
        self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)  
        self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
        self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)
        self.final_Conv = Conv3Relu(inplanes * 4, inplanes)

        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)

        inter_channels = inplanes // 4
        self.out_Conv = nn.Sequential(Conv3Relu(inplanes, inter_channels),
                                  nn.Dropout(0.2),  
                                  nn.Conv2d(inter_channels, num_class, (1, 1)))

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        fa1 = self.bsde1(fa1)
        fb1 = self.bsde1(fb1)
        fa2 = self.bsde2(fa2)
        fb2 = self.bsde2(fb2)
        fa3 = self.bsde3(fa3)
        fb3 = self.bsde3(fb3)
        fa4 = self.bsde4(fa4)
        fb4 = self.bsde4(fb4)

        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4]) 

        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 2
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 8

        change4 = self.expand_field(change4)

        change3_2 = self.stage4_Conv_after_up(self.up(change4))
        change3 = self.stage3_Conv2(torch.cat([change3, change3_2], 1))
        change2_2 = self.stage3_Conv_after_up(self.up(change3))
        change2 = self.stage2_Conv2(torch.cat([change2, change2_2], 1))
        change1_2 = self.stage2_Conv_after_up(self.up(change2))
        change1 = self.stage1_Conv2(torch.cat([change1, change1_2], 1))

        change4 = self.gfa41(change1, self.up8(self.stage4_Conv3(change4)))
        change3 = self.gfa31(change1, self.up4(self.stage3_Conv3(change3)))
        change2 = self.gfa21(change1, self.up(self.stage2_Conv3(change2)))

        [change1, change2, change3, change4] = self.drop([change1, change2, change3, change4]) 

        change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1))
        change = self.out_Conv(change)

        return change

