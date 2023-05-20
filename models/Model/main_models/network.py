import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *

from models.Model.do_conv_pytorch import DOConv2d
from models.Model.main_models.SLDD import SLDD
from models.Model.DifferentialFocus import DF_block
from models.Model.AxialSemanticEnhancenment import ASE_block
# from Visual import show_feature_map

import matplotlib.pyplot as plt


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.5):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            DOConv2d(in_channels, inter_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            DOConv2d(inter_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.block(x)


class SLDDNet(nn.Module):
    def __init__(self, backbone='SLDD', drop=0.1):
        super(SLDDNet, self).__init__()
        assert backbone in ['resnet34', 'SLDD']

        if backbone == 'resnet34':  # Structure changes required
            self.backbone = resnet34(pretrained=True)
        elif backbone == 'SLDD':
            self.backbone = SLDD(img_size=256, in_chans=3)
        else:
            raise NotImplementedError

        self.head = _FCNHead(96, 2, drop=drop)

        self.DF3 = DF_block(channels=216, out_channels=216, r=8)
        self.DF2 = DF_block(channels=216, out_channels=176)
        self.DF1 = DF_block(channels=176, out_channels=96)

        self.ASE3 = ASE_block(dim=216, key_dim=16, num_heads=8)
        self.ASE2 = ASE_block(dim=176, key_dim=16, num_heads=8)
        self.ASE1 = ASE_block(dim=96, key_dim=16, num_heads=8)

        self.fuse_conv = nn.Sequential(
            DOConv2d(432, 216, kernel_size=1, bias=False),
            nn.BatchNorm2d(216),
            nn.ReLU()
        )

        self.skip2_conv = nn.Sequential(
            DOConv2d(352, 176, kernel_size=1, bias=False),
            nn.BatchNorm2d(176),
            nn.ReLU()
        )

        self.skip1_conv = nn.Sequential(
            DOConv2d(192, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.num_images = 0

        self.sigmoid = nn.Sigmoid()
        self.loss4 = DOConv2d(216, 2, 1)
        self.loss3 = DOConv2d(216, 2, 1)
        self.loss2 = DOConv2d(176, 2, 1)
        self.loss1 = DOConv2d(96, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, img_name="None"):

        # y = x

        lay1list = []
        lay2list = []
        outlist = []
        _, _, hei, wid = x.shape

        out1 = self.backbone(x)
        c1, c2, c3, c4 = out1
        out2 = self.backbone(y)
        a1, a2, a3, a4 = out2

        # Deep supervision---add--

        c1 = self.ASE1(c1)
        c2 = self.ASE2(c2)
        c3 = self.ASE3(c3)
        c4 = self.ASE3(c4)

        a1 = self.ASE1(a1)
        a2 = self.ASE2(a2)
        a3 = self.ASE3(a3)
        a4 = self.ASE3(a4)

        # ----------------------Fusion-------------------------
        totalout = torch.cat((c4, a4), dim=1)
        totalout = self.fuse_conv(totalout)

        # Deep supervision---add--

        # ----------------------------------------------------------
        skip3 = torch.cat((c3, a3), dim=1)
        skip3 = self.fuse_conv(skip3)

        totalout = self.DF3(totalout, skip3)

        # Deep supervision---add--

        # -----------------------------------------------------------
        skip2 = torch.cat((c2, a2), dim=1)
        skip2 = self.skip2_conv(skip2)

        totalout = F.interpolate(totalout, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)

        totalout = self.DF2(totalout, skip2)

        # Deep supervision---add--

        # -----------------------------------------------------------
        skip1 = torch.cat((c1, a1), dim=1)
        skip1 = self.skip1_conv(skip1)

        totalout = F.interpolate(totalout, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)

        totalout = self.DF1(totalout, skip1)

        # Deep supervision---add--

        # ------------------------------------------------------------
        pred = self.head(totalout)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)
        out = self.sigmoid(out)

        # hotmap
        # self.vis_fearure(hotmaplist, img_name)
        # show_feature_map()

        outlist.append(out)
        return outlist, lay1list, lay2list