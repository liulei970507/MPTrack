from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import Block
from lib.models.layers.attn_blocks import CASTBlock
from timm.models.layers import DropPath, Mlp
class MultiHeadCrossAttention(nn.Module):
    """
    一个标准的多头交叉注意力模块。
    Query 来自一个输入序列 (x)，而 Key 和 Value 来自另一个输入序列 (context)。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 分别为 Q, K, V 创建线性投影层
        # 注意：Q 来自一个源，K/V 来自另一个源
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, mask=None):
        """
        前向传播函数。
        
        Args:
            x (torch.Tensor): 查询序列 (Query)，形状为 (B, N_q, C)，其中 B 是批量大小, N_q 是查询序列长度, C 是特征维度。
            context (torch.Tensor): 键/值序列 (Key/Value)，形状为 (B, N_kv, C)，其中 N_kv 是键/值序列长度。
            mask (torch.Tensor, optional): 注意力掩码。
        
        Returns:
            torch.Tensor: 输出张量，形状为 (B, N_q, C)。
        """
        B, N_q, C = x.shape
        _, N_kv, _ = context.shape

        # 1. 线性投影
        # q 来自 x, k 和 v 来自 context
        q = self.q_proj(x)  # (B, N_q, C)
        k = self.k_proj(context) # (B, N_kv, C)
        v = self.v_proj(context) # (B, N_kv, C)

        # 2. Reshape 以实现多头注意力
        # (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 3. 计算注意力分数
        # (B, H, N_q, D_h) @ (B, H, D_h, N_kv) -> (B, H, N_q, N_kv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4. 用注意力分数加权 Value
        # (B, H, N_q, N_kv) @ (B, H, N_kv, D_h) -> (B, H, N_q, D_h)
        x = (attn @ v)

        # 5. Reshape 回原始维度
        # (B, H, N_q, D_h) -> (B, N_q, H, D_h) -> (B, N_q, C)
        x = x.permute(0, 2, 1, 3).reshape(B, N_q, C)
        
        # 6. 最终的线性投影和 Dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

### 第2步：构建 `Cross_Block`

class Cross_Block(nn.Module):
    """
    一个使用多头交叉注意力的 Transformer 编码器块。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = MultiHeadCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context):
        """
        前向传播函数。
        
        Args:
            x (torch.Tensor): 查询序列，形状为 (B, N_q, C)。
            context (torch.Tensor): 键/值序列，形状为 (B, N_kv, C)。
        
        Returns:
            torch.Tensor: 输出张量，形状为 (B, N_q, C)。
        """
        # 交叉注意力部分
        # x 经过 Norm 成为 Query
        # context 经过 Norm 成为 Key 和 Value
        attn_output = self.attn(self.norm1_q(x), self.norm1_kv(context))
        
        # 第一个残差连接 (x + DropPath(Attention Output))
        x = x + self.drop_path(attn_output)
        
        # MLP 部分
        mlp_output = self.mlp(self.norm2(x))
        
        # 第二个残差连接 (x + DropPath(MLP Output))
        x = x + self.drop_path(mlp_output)
        
        return x
    
class ZFuse_wsq(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, len_z=-1):
        super().__init__()

        
        # self.t_fusion = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            # attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)
        # self.z_fusion_rgb = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)
        
        # self.z_fusion_tir = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)
        
        
        self.specific2rgb = Cross_Block(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                    attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
                )
        
        self.specific2tir = Cross_Block(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                    attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer,
                )
        
        
    def forward(self, x_v, x_i, lens_z):
        # fused_t = self.t_fusion(torch.cat([x_v[:, lens_z*2:lens_z*3, :], x_i[:, lens_z*2:lens_z*3, :]], dim=1))  # [B, 64, C]
        # fused_t_rgb = fused_t[:, :lens_z, :]
        # fused_t_tir = fused_t[:, lens_z:, :]

        # 模态内self-att
        # temp_x_v = self.z_fusion_rgb(torch.cat([x_v[:, :lens_z*3, :],fused_t], dim=1))[:, :lens_z*3, :]
        # temp_x_i = self.z_fusion_tir(torch.cat([x_i[:, :lens_z*3, :],fused_t], dim=1))[:, :lens_z*3, :]
        # 模态间互增强，只用判别性最高的specific模板
        xv = self.specific2rgb(x_v[:, :lens_z*3, :], torch.cat([x_v[:, :lens_z*3, :], x_i[:, lens_z*2:lens_z*3, :]], dim=1))
        xi = self.specific2tir(x_i[:, :lens_z*3, :], torch.cat([x_i[:, :lens_z*3, :], x_v[:, lens_z*2:lens_z*3, :]], dim=1))
        
        return xv, xi
