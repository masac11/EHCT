
# from visualizer import get_local
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from videoanalyst.model.transfor.transfor_base import (TRACK_TRANSFOR, VOS_TRANSFOR)
from videoanalyst.model.module_base import ModuleBase
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # inner_dim = dim_head *  heads
        # project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

    # @get_local('attn_map')
    def forward(self, x):
        x_for_qkv = x
        B, N, C = x.shape
        q_linear_out = self.q_linear(x_for_qkv)
        q = q_linear_out.reshape(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k = k_linear_out.reshape(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v = v_linear_out.reshape(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3).contiguous()

        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn_map = attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # return self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

@VOS_TRANSFOR.register
@TRACK_TRANSFOR.register
class ViT(ModuleBase):
    def __init__(self, *, pretrain_img_size=280, patch_size=8, dim=96, depth=4, heads=4, mlp_dim=384, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_size = pretrain_img_size
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

        self.convert_3 = nn.Conv2d(3 * 5 * 2, 3, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(size=(pretrain_img_size, pretrain_img_size), mode='bilinear', align_corners=True)
        self.conv_f = nn.Conv2d(96, 256, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, img_pos, img_neg):
        x = self.convert_3(torch.cat([img_pos[0], img_pos[1], img_pos[2], img_pos[3], img_pos[4], \
                                      img_neg[0], img_neg[1], img_neg[2], img_neg[3],
                                      img_neg[4]], dim=1))
        x = self.upsample(x)
        x = self.to_patch_embedding(x)
        b, n, c = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # Todo: window attention 
        x = self.transformer(x)
        # h,w = image_size//patch_size, image_size//patch_size
        x = x.view(b, 35, 35, c)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn(self.conv_f(x)))
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return None, x