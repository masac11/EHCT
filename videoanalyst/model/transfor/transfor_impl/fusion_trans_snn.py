import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.transfor.transfor_base import (TRACK_TRANSFOR, VOS_TRANSFOR)
from spikingjelly.activation_based import neuron, functional, layer

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x, s_x):
        T,B,N,C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        s_x_for_kv = s_x.flatten(0, 1)
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        # v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        # x = self.attn_lif(x)
        x = x.flatten(0, 1)
        # x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        x = self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, s_x):
        x = x + self.attn(x, s_x)
        x = x + self.mlp(x)
        return x

class PosEmd(nn.Module):
    def __init__(self, img_size_h=16, img_size_w=16, in_channels=256, embed_dims=256, patch_size=1):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.embed_dims = embed_dims

        self.pos_embedding = nn.Parameter(torch.randn(5, 1, embed_dims, self.H, self.W))
        self.conv = layer.Conv2d(in_channels=256,out_channels=embed_dims, kernel_size=patch_size,
                                  stride=patch_size)
        # self.conv = layer.Conv2d(in_channels=in_channels, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        T, B, _, H, W = x.shape
        x = self.conv(x)
        x = x + self.pos_embedding
        x = x.permute(0, 1, 3, 4, 2).view(T, B, self.num_patches, self.embed_dims)
        return x
    
@VOS_TRANSFOR.register
@TRACK_TRANSFOR.register
class FusionTransSNN(ModuleBase):
    def __init__(self,
                 pretrain_img_size=224,
                 img_size_h=16, img_size_w=16, in_channels=256,
                 embed_dims=256, num_heads=4, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=3, sr_ratios=1, T = 5
                 ):
        super().__init__()
        self.T = T  # time step
        self.depths = depths
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        pos_embed = PosEmd(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])
        self.block = block

        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        self.apply(self._init_weights)
        out_channels = 256
        self.conv_f_snn_out = nn.Conv2d(in_channels=embed_dims, out_channels=out_channels, kernel_size=6, stride=1, padding=0)
        self.conv_f_trans_out = nn.Conv2d(in_channels=embed_dims, out_channels=out_channels, kernel_size=6, stride=1, padding=0)
        self.conv_s_snn_out = nn.Conv2d(in_channels=embed_dims, out_channels=out_channels, kernel_size=1)
        self.conv_s_trans_out = nn.Conv2d(in_channels=embed_dims, out_channels=out_channels, kernel_size=1)
        self.bn_snn_out = nn.BatchNorm2d(out_channels)
        self.bn_f_trans_out = nn.BatchNorm2d(out_channels)
        self.bn_s_snn_out = nn.BatchNorm2d(out_channels)
        self.bn_s_trans_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.convert_64_dim = nn.Conv2d(in_channels=256, out_channels=embed_dims, kernel_size=1)
        self.convert_128_dim = nn.Conv2d(in_channels=256, out_channels=embed_dims, kernel_size=1)
        self.convert_256_dim = nn.Conv2d(in_channels=256, out_channels=embed_dims, kernel_size=1)
        self.convert_dim_64 = nn.Conv2d(in_channels=embed_dims, out_channels=256, kernel_size=1)
        self.convert_dim_128 = nn.Conv2d(in_channels=embed_dims, out_channels=256, kernel_size=1)
        self.convert_dim_256 = nn.Conv2d(in_channels=embed_dims, out_channels=256, kernel_size=1)
        self.layer_norm1 = nn.LayerNorm(embed_dims)
        self.layer_norm2 = nn.LayerNorm(embed_dims)
        self.layer_norm3 = nn.LayerNorm(embed_dims)
        self.batch_norm64 = nn.BatchNorm2d(256)
        self.batch_norm128 = nn.BatchNorm2d(256)

        self.snn_block1 = nn.Sequential(
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            layer.BatchNorm2d(256),
            neuron.LIFNode(store_v_seq=True),
        )
        self.snn_block2 = nn.Sequential(
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            layer.BatchNorm2d(256),
            neuron.LIFNode(store_v_seq=True),
        )
        self.snn_block3 = nn.Sequential(
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            layer.BatchNorm2d(256),
            neuron.LIFNode(store_v_seq=True), 
        )

        self.conv1x1_fusion1 = nn.Conv2d(256*2, 256, kernel_size=1, bias=True)
        self.conv1x1_fusion2 = nn.Conv2d(256*2, 256, kernel_size=1, bias=True)
        self.conv1x1_fusion3 = nn.Conv2d(256*2, 256, kernel_size=1, bias=True)

        self.fusion1_lif = neuron.LIFNode(store_v_seq=True)
        self.fusion2_lif = neuron.LIFNode(store_v_seq=True)
        self.fusion3_lif = neuron.LIFNode(store_v_seq=True)

        functional.set_step_mode(self, step_mode='m')

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def resize(self, x, x_h, target_h):
        if x_h > target_h:
            return F.adaptive_avg_pool2d(x, output_size=(target_h, target_h))
        return F.interpolate(x, size=(target_h, target_h)) 
    
    def forward_features(self, x, snn_input):
        T, B, C, H, W = snn_input.shape
        tran_h = 16
        block = getattr(self, f"block")
        pos_embed = getattr(self, f"pos_embed")
        x = pos_embed(x)
        attn_blk1, attn_blk2, attn_blk3  = block[0], block[1], block[2]

        # snn1
        t = x.reshape(T*B, tran_h, tran_h, self.embed_dims).permute(0,3,1,2)
        fu = self.conv1x1_fusion1(torch.cat((t, snn_input.reshape(T*B, self.embed_dims, tran_h, tran_h)), dim=1))
        fu = fu.reshape(T, B, t.shape[1], t.shape[2], t.shape[3])
        fu = self.fusion1_lif(fu)
        s1_out = self.snn_block1(fu)
        s1_mem = self.snn_block1[-1].v_seq

        # trans1 snn1 -> trans1
        s_x = s1_mem
        x = attn_blk1(x, s_x.flatten(-2).permute(0,1,3,2))

        # snn2 trans1 -> snn1
        trans1_out = x.reshape(T*B, tran_h, tran_h, self.embed_dims).permute(0,3,1,2)
        fu = self.conv1x1_fusion2(torch.cat((trans1_out, s1_out.reshape(T*B, self.embed_dims, tran_h, tran_h)), dim=1))
        fu = fu.reshape(T, B, trans1_out.shape[1], trans1_out.shape[2], trans1_out.shape[3])
        fu = self.fusion2_lif(fu)
        s2_out = self.snn_block2(fu)
        s2_mem = self.snn_block2[-1].v_seq

        # trans2 snn2 -> trans2
        s_x = s2_mem
        x = attn_blk2(x, s_x.flatten(-2).permute(0,1,3,2))

        # snn3 trans2 -> snn3
        trans2_out = x.reshape(T*B, tran_h, tran_h, self.embed_dims).permute(0,3,1,2)
        fu = self.conv1x1_fusion3(torch.cat((trans2_out, s2_out.reshape(T*B, self.embed_dims, tran_h, tran_h)), dim=1))
        fu = fu.reshape(T, B, trans2_out.shape[1], trans2_out.shape[2], trans2_out.shape[3])
        fu = self.fusion3_lif(fu)
        s3_out = self.snn_block3(fu)
        s3_mem = self.snn_block3[-1].v_seq

        # trans3 snn3 -> trans3
        s_x = s3_mem
        x = attn_blk3(x, s_x.flatten(-2).permute(0,1,3,2))

        # s3_out = torch.div(torch.sum(self.snn_block3[-1].v_seq, dim=0), T) 
        return x, s3_mem
    
    def forward(self, x, first=False):
        T, B, C, H, W = x.shape

        if first:
            functional.reset_net(self)
        
        trans_size = 16
        snn_input = x.clone()
        trans_out, snn_out = self.forward_features(x, snn_input)
        snn_out = snn_out.mean(0)
        trans_out = trans_out.view(T, B, trans_size, trans_size, -1)
        trans_out = trans_out.mean(0)
        trans_out = trans_out.permute(0, 3, 1, 2).contiguous()
        if first:
            snn_out = self.relu(self.bn_snn_out(self.conv_f_snn_out(snn_out)))
            trans_out = self.relu(self.bn_f_trans_out(self.conv_f_trans_out(trans_out)))
            functional.reset_net(self)
            return snn_out, trans_out
        
        trans_out = F.interpolate(trans_out, size=(33, 33))
        snn_out = F.interpolate(snn_out, size=(33, 33))
        snn_out = self.relu(self.bn_s_snn_out(self.conv_s_snn_out(snn_out)))
        trans_out = self.relu(self.bn_s_trans_out(self.conv_s_trans_out(trans_out)))
        functional.reset_net(self)
        return snn_out, trans_out