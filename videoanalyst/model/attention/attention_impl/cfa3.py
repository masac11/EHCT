import torch
import torch.nn as nn
from torchvision.models import resnet34
from timm.models.layers import DropPath
from videoanalyst.model.attention.attention_base import (TRACK_ATTENTION, VOS_ATTENTION)
from videoanalyst.model.module_base import ModuleBase
import torch.nn.functional as F

@VOS_ATTENTION.register
@TRACK_ATTENTION.register
class CrossAttention3(ModuleBase):
    "Implementation of self-attention"

    def __init__(self, dim=256, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=6, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        

    def forward(self, f_kv, x, f3):
        x0 = f3.mean(0)
        _, _, H, W = x.shape
        if H == 11:
            x0 = self.relu(self.bn1(self.conv1(x0)))
        else:
            x0 = self.relu(self.bn2(F.interpolate(x0, size=(33, 33))))
        
        x_b, x_c, x_h, x_w = x.shape
        x = x.flatten(2).transpose(2,1)
        f_kv = f_kv.flatten(2).transpose(2,1)
        B, N, C = x.shape
        # print(x.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(f_kv).reshape(B, N, 2, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
#         print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(2,1).reshape(x_b, x_c, x_h, x_w).contiguous()
        out = out + x0
        return out