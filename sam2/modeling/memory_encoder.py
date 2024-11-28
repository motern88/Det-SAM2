# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d


class MaskDownSampler(nn.Module):
    """
    逐步将掩码通过total_stride进行下采样，每次下采样的步长为stride。
    注意，LayerNorm是按*token*应用的，就像在ViT中一样。

    每次下采样（因子为stride**2），通道容量就会增加相同的因子。
    最后，我们通过线性投影到embed_dim个通道。
    """

    def __init__(
        self,
        embed_dim=256,  # 嵌入维度
        kernel_size=4,  # 卷积核大小
        stride=4,  # 步长
        padding=0,  # 填充
        total_stride=16,  # 总步长
        activation=nn.GELU,  # 激活函数
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))  # 计算需要的层数
        assert stride**num_layers == total_stride  # 确保总步幅是步幅的幂
        self.encoder = nn.Sequential()  # 定义顺序容器
        mask_in_chans, mask_out_chans = 1, 1  # 初始化掩码的输入和输出通道数
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)  # 更新掩码的输出通道数
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )  # 添加卷积层
            self.encoder.append(LayerNorm2d(mask_out_chans))  # 添加LayerNorm层
            self.encoder.append(activation())  # 添加激活函数
            mask_in_chans = mask_out_chans  # 更新输入通道数

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))  # 最后的线性投影层

    def forward(self, x):
        return self.encoder(x)  # 前向传播


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. 有两个等效的实现：
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv；全部在 (N, C, H, W)
    (2) DwConv -> Permute到 (N, H, W, C)；LayerNorm (channels_last) -> Linear -> GELU -> Linear；再Permute回
    我们使用 (2) 因为我们发现它在PyTorch中稍微快一点

    参数：
        dim (int): 输入通道数。
        drop_path (float): 随机深度率。默认值：0.0
        layer_scale_init_value (float): 层缩放的初始化值。默认值：1e-6。
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # 深度卷积
        self.norm = LayerNorm2d(dim, eps=1e-6)  # LayerNorm层
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # 点卷积/1x1卷积，使用线性层实现
        self.act = nn.GELU()  # 激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 第二个线性层
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )  # 层缩放参数
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # 随机深度

    def forward(self, x):
        input = x  # 保存输入以用于残差连接
        x = self.dwconv(x)  # 深度卷积
        x = self.norm(x)  # LayerNorm
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)  # 点卷积
        x = self.act(x)  # 激活函数
        x = self.pwconv2(x)  # 第二个点卷积
        if self.gamma is not None:
            x = self.gamma * x  # 应用层缩放
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)  # 残差连接和随机深度
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()  # 默认为恒等映射
        self.layers = get_clones(layer, num_layers)  # 复制层以构建多个层

        if input_projection:
            assert dim is not None  # 确保提供了dim
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)  # 输入投影层

    def forward(self, x):
        # 通常情况下 x: (N, C, H, W)
        x = self.proj(x)  # 应用输入投影
        for layer in self.layers:
            x = layer(x)  # 逐层前向传播
        return x


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # pix_feats的输入维度
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler  # 掩码下采样器

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)  # 像素特征的投影层
        self.fuser = fuser  # 特征融合器
        self.position_encoding = position_encoding  # 位置编码
        self.out_proj = nn.Identity()  # 默认为恒等映射
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)  # 如果输出维度不同，使用卷积层进行投影

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## 处理掩码
        # 使用sigmoid，使其与真实掩码（布尔值）之间的领域偏移减少
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)  # 应用sigmoid函数
        masks = self.mask_downsampler(masks)  # 下采样掩码

        ## 融合pix_feats和下采样的掩码
        # 如果视觉特征在CPU上，将其转换到CUDA
        pix_feat = pix_feat.to(masks.device)  # 将像素特征移动到掩码的设备上

        x = self.pix_feat_proj(pix_feat)  # 像素特征投影
        x = x + masks  # 融合像素特征和掩码
        x = self.fuser(x)  # 通过特征融合器
        x = self.out_proj(x)  # 线性投影到输出维度

        pos = self.position_encoding(x).to(x.dtype)  # 计算位置编码

        return {"vision_features": x, "vision_pos_enc": [pos]}  # 返回视觉特征和位置编码
