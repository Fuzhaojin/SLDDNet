import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Model.do_conv_pytorch import DOConv2d


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Att(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(Att, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = DOConv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = DOConv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = DOConv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.att = Att(in_channels, in_channels)
        self.conv_1x1 = DOConv2d(out_channels, out_channels, 1, stride=1, padding=0)
        self.conv2_1x1 = DOConv2d(in_channels, out_channels, 1, stride=1, padding=0)
        self.outChannel = out_channels

    def forward(self, x, fre=True):
        feat = self.att(x)
        b,c,h,w = feat.size()
        if c != self.outChannel:
            feat = self.conv2_1x1(feat)

        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten

class DF_block(nn.Module):
    def __init__(self, channels, out_channels, r=16):
        super().__init__()
        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.spatial_mlp = nn.Sequential(nn.Linear(r*r, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
        self.feat1_att = ChannelAtt(channels, out_channels)
        self.context_mlp = nn.Sequential(*[nn.Linear(r*r, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)])
        self.feat2_att = ChannelAtt(out_channels, out_channels)
        self.context_head = DOConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.smooth = DOConv2d(out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, feat1, feat2):
        att_feat1, att1 = self.feat1_att(feat1)
        att_feat2, att2 = self.feat2_att(feat2)

        b, c, h, w = att1.size()
        att1_split = att1.view(b, self.r, c // self.r)
        att2_split = att2.view(b, self.r, c // self.r)

        chl_affinity = torch.bmm(att1_split, att2_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)

        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))

        re_att1 = torch.sigmoid(att1 + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_att2 = torch.sigmoid(att2 + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))

        att_feat2 = torch.mul(att_feat2, re_att2)
        att_feat1 = torch.mul(att_feat1, re_att1)

        att_feat2 = F.interpolate(att_feat2, att_feat1.size()[2:], mode='bilinear', align_corners=False)
        att_feat2 = self.context_head(att_feat2)

        out = self.smooth(att_feat1 + att_feat2)
        return out


if __name__ == '__main__':
    x = torch.randint(1, 10, (1, 176, 128, 128)).float()
    y = torch.randint(1, 10, (1, 96, 128, 128)).float()
    net = DF_block(channels=176, out_channels=96, r=8)
    out = net(x, y)
    print(out.shape)
