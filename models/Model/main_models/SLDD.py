import torch
from torch import einsum, nn
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import math
import numpy as np
from einops import rearrange

from timm.models.layers import DropPath, trunc_normal_
from mmcv.runner import (BaseModule, load_checkpoint)
from models.Model.do_conv_pytorch import DOConv2d
from models.Model.ChannelAttention import CA

class DConv(nn.Module):
    def __init__(self, dim=96):
        super(DConv, self).__init__()
        self.dconv = DOConv2d(dim, dim, 3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dconv = DConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, size):
        H, W = size
        x = self.fc1(x)
        x = self.act(x + self.dconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = DOConv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
    ):
        super().__init__()

        self.dconv = DOConv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=out_ch,
            bias=False
        )
        # linear
        self.pconv = DOConv2d(out_ch, out_ch, 1, stride=1, padding=0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class DOC_patchembed(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )

    def forward(self, x):

        x = self.patch_conv(x)

        return x


class PosE(nn.Module):
    def __init__(self, dim, k=3):
        super(PosE, self).__init__()

        self.proj = DOConv2d(dim, dim, k, stride=1, padding=k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        return x


class PSFS(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(PSFS, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.ConvLinear = Conv2d_BN(6 * inter_planes, out_planes, kernel_size=1, stride=1, act_layer=None)
        self.shortcut = Conv2d_BN(in_planes, out_planes, kernel_size=1, stride=stride, act_layer=None)
        self.relu = nn.ReLU(inplace=False)
        self.att = CA(inp=out_planes, oup=out_planes)

        self.conv33 = nn.Sequential(
            Conv2d_BN(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, act_layer=None),
            Conv2d_BN(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, pad=1, groups=groups, act_layer=nn.ReLU),
            Conv2d_BN(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, pad=vision + 1, dilation=vision + 1, act_layer=None, groups=groups),
        )

        self.conv55 = nn.Sequential(
            Conv2d_BN(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, act_layer=None),
            Conv2d_BN(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, pad=1, groups=groups, act_layer=nn.ReLU),
            Conv2d_BN(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, pad=vision + 2, dilation=vision + 2, act_layer=None, groups=groups)
        )

        self.conv77 = nn.Sequential(
            Conv2d_BN(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, act_layer=None),
            Conv2d_BN(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, pad=1, groups=groups, act_layer=nn.ReLU),
            Conv2d_BN((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, pad=1, groups=groups, act_layer=nn.ReLU),
            Conv2d_BN(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, pad=vision + 4, dilation=vision + 4, act_layer=None, groups=groups)
        )

    def forward(self, x):
        x33 = self.conv33(x)
        x55 = self.conv55(x)
        x77 = self.conv77(x)
        out = torch.cat((x33, x55, x77), 1)
        out = self.ConvLinear(out)
        out = self.att(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class SemanticAggregation(nn.Module):

    def __init__(self, embed_dim, num_path=4, isPool=False, stage=0):
        super(SemanticAggregation, self).__init__()

        if stage == 3:
            self.patch_embeds = nn.ModuleList([
                DOC_patchembed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=4 if (isPool and idx == 0) or (stage > 1 and idx == 1) else 2,  # else 原始为1
                ) for idx in range(num_path + 1)
            ])
        else:

            self.patch_embeds = nn.ModuleList([
                DOC_patchembed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if (isPool and idx == 0) or (stage > 1 and idx == 1) else 1,
                ) for idx in range(num_path + 1)
            ])

    def forward(self, x):
        att_list = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_list.append(x)

        return att_list


class GSSS(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feat):
        h, w = feat.size(2), feat.size(3)
        feat = torch.split(feat, self.channel_splits, dim=1)
        priors = [F.upsample(input=self.stages[i](feat[i]), size=(h, w), mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)
        
        return self.relu(bottle)
    

class SemanticSelector(nn.Module):
    def __init__(self, Ch, h, window):

        super().__init__()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = DOConv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]
        self.gsss = GSSS(Ch * h)

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size

        v_img = rearrange(v, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        gsss = self.gsss(v_img)

        # LSSS
        v_list = torch.split(v_img, self.channel_splits, dim=1)
        LSSS_list = [
            conv(x) for conv, x in zip(self.conv_list, v_list)
        ]
        lsss = torch.cat(LSSS_list, dim=1)

        lsss = rearrange(lsss, "B (h Ch) H W -> B h (H W) Ch", h=h)
        gsss = rearrange(gsss, "B (h Ch) H W -> B h (H W) Ch", h=h)

        dynamic_filters = q * lsss + gsss
        return dynamic_filters


class Seatt_semantic_selector(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            semantic_strength_selector=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sss = semantic_strength_selector  # semantic strength selector

    def forward(self, x, size):
        B, N, C = x.shape

        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_softmax = k.softmax(dim=2)

        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)

        crpe = self.sss(q, v, size=size) # semantic strength selector

        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TSSBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            drop_path=0.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            semantic_strength_selector=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.pe = shared_cpe
        self.sss = semantic_strength_selector
        self.att_result = Seatt_semantic_selector(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            semantic_strength_selector=semantic_strength_selector,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, size):
        if self.pe is not None:
            x = self.pe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.att_result(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur, size))
        return x


class TransformerSemanticSelector(nn.Module):
    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            drop_path_list=[],
            qk_scale=None,
            crpe_window={
                3: 2,
                5: 3,
                7: 3
            },
    ):
        super().__init__()

        self.num_layers = num_layers
        self.pe = PosE(dim, k=3)
        self.sss = SemanticSelector(Ch=dim // num_heads,
                                  h=num_heads,
                                  window=crpe_window)
        self.TSS_layers = nn.ModuleList([
            TSSBlock(
                dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_list[idx],
                qk_scale=qk_scale,
                shared_cpe=self.pe,
                semantic_strength_selector=self.sss,
            ) for idx in range(self.num_layers)
        ])

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        for layer in self.TSS_layers:
            x = layer(x, (H, W))

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class FeatureReconstruction(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = in_features // 2
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv1 = Conv2d_BN(in_features, hidden_features, act_layer=act_layer)
        self.dconv = DOConv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=False, groups=hidden_features)
        self.conv2 = Conv2d_BN(hidden_features, out_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.dconv(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)

        return res + out


class PM(nn.Module):
    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            num_path=4,
            drop_path_list=[],
            id_stage=0,
    ):
        super().__init__()

        self.FR = FeatureReconstruction(in_features=embed_dim, out_features=embed_dim)
        self.PSFS = PSFS(in_planes=embed_dim,out_planes=embed_dim)

        if id_stage > 0:
            self.aggregate = Conv2d_BN(embed_dim * (num_path),
                                       out_embed_dim,
                                       act_layer=nn.Hardswish)
            self.transformer_semantic_selector = nn.ModuleList([
                TransformerSemanticSelector(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                ) for _ in range(num_path)
            ])
        else:
            self.aggregate = Conv2d_BN(embed_dim * (num_path),
                                       out_embed_dim,
                                       act_layer=nn.Hardswish)

    def forward(self, inputs, id_stage):

        if id_stage > 0:
            inputs[0] = self.PSFS(inputs[0])

            att_outputs = [self.FR(inputs[0])]
            for x, encoder in zip(inputs[1:], self.transformer_semantic_selector):
                _, _, H, W = x.shape

                x = x.flatten(2).transpose(1, 2)
                att_outputs.append(encoder(x, size=(H, W)))

            for i in range(len(att_outputs)):
                if att_outputs[i].shape[2:] != att_outputs[0].shape[2:]:
                    att_outputs[i] = F.interpolate(att_outputs[i], size=att_outputs[0].shape[2:], mode='bilinear',
                                                   align_corners=True)
            out_concat = att_outputs[0] + att_outputs[1]
        else:
            out_concat = self.FR(inputs[0] + inputs[1])

        out = self.aggregate(out_concat)

        return out

def select_drp(drop_path_rate, num_layers, maxI):

    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(maxI):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr



class SLDD(BaseModule):
    def __init__(
        self,
        maxI=4,
        num_path=[1, 1, 1, 1],
        num_layers=[1, 2, 6, 2],
        embed_dims=[32, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        img_size=256,
        drop_path_rate=0.0,
        in_chans=3,
        num_classes=1000,
        strides=[4, 2, 2, 2],
        pretrained=None, init_cfg=None,
    ):
        super().__init__()
        if isinstance(pretrained, str):
            self.init_cfg = pretrained
        self.num_classes = num_classes
        self.maxI = maxI

        dpr = select_drp(drop_path_rate, num_layers, maxI)

        self.Parallel_Model = nn.ModuleList([
            PM(
                embed_dims[i],
                embed_dims[i + 1]
                if not (i + 1) == self.maxI else embed_dims[i],
                num_layers[i],
                num_heads[i],
                mlp_ratios[i],
                num_path[i],
                drop_path_list=dpr[i],
                id_stage=i,
            ) for i in range(self.maxI)
        ])

        self.feature_pre_processing = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=1,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        self.SIA = nn.ModuleList([
            SemanticAggregation(
                embed_dims[i],
                num_path=num_path[i],
                isPool=True if i == 1 else False,
                stage=i,
            ) for i in range(self.maxI)
        ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, DOConv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        if isinstance(self.init_cfg, str):
            pass
        else:
            self.apply(self._init_weights)

    def forward(self, x):
        list = []
        x = self.feature_pre_processing(x)
        for i in range(self.maxI):
            inputs = self.SIA[i](x)
            x = self.Parallel_Model[i](inputs, i)
            list.append(x)
            
        return list