import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HAFE import HAFE
from models.block.Base import ChannelChecker
from models.CFFI import CFFI
from collections import OrderedDict
from util.common import ScaleInOutput

class HATNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = opt.inplanes
        self.hafe = HAFE()
        self.check_channels = ChannelChecker(self.hafe, self.inplanes, opt.input_size)
        self.cffi = CFFI(self.inplanes, 2)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   

    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."
        fa1, fa2, fa3, fa4 = self.hafe(xa) 
        fa1, fa2, fa3, fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, fb4 = self.hafe(xb)
        fb1, fb2, fb3, fb4 = self.check_channels(fb1, fb2, fb3, fb4)
        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4  
        change = self.cffi(ms_feats)
        out_size=(h_input, w_input)
        out = F.interpolate(change, size=out_size, mode='bilinear', align_corners=True)
        return out
   
    def _init_weight(self, pretrain=''): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)

class EnsembleHATNet(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=448):
        super(EnsembleHATNet, self).__init__()
        self.method = method
        self.models_list = []
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                ckp_path = os.path.join(ckp_path, weight_file[0])
            print("--Load model: {}".format(ckp_path))
            model = torch.load(ckp_path, map_location=device)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb):
        xa, xb = self.scale.scale_input((xa, xb))
        out1 = 0
        cd_pred = None

        for i, model in enumerate(self.models_list):
            outs = model(xa, xb)
            if not isinstance(outs, tuple):
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))
                out1 += outs[0]
                _, cd_pred = torch.max(out1, 1)

        return cd_pred