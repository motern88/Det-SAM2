# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""一些用于主干网络的工具，特别是用于窗口化"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    将输入划分为不重叠的窗口，如果需要则进行填充。
    参数:
        x (tensor): 输入张量，形状为 [B, H, W, C]。
        window_size (int): 窗口大小。
    返回:
        windows: 分割后的窗口，形状为 [B * num_windows, window_size, window_size, C]。
        (Hp, Wp): 分割前的填充后的高度和宽度。
    """
    B, H, W, C = x.shape  # 获取输入的形状

    pad_h = (window_size - H % window_size) % window_size  # 计算高度需要的填充量
    pad_w = (window_size - W % window_size) % window_size  # 计算宽度需要的填充量
    if pad_h > 0 or pad_w > 0:  # 如果需要填充
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 进行填充
    Hp, Wp = H + pad_h, W + pad_w  # 计算填充后的高度和宽度

    # 将输入张量视为多个小窗口并重新排列
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)  # 返回窗口化后的张量和填充后的尺寸


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    将窗口恢复为原始序列，并移除填充。
    参数:
        x (tensor): 输入张量，形状为 [B * num_windows, window_size, window_size, C]。
        window_size (int): 窗口大小。
        pad_hw (Tuple): 填充后的高度和宽度 (Hp, Wp)。
        hw (Tuple): 填充前的原始高度和宽度 (H, W)。
    返回:
        x: 取消分割后的序列，形状为 [B, H, W, C]。
    """
    Hp, Wp = pad_hw  # 获取填充后的高度和宽度
    H, W = hw  # 获取填充前的原始高度和宽度
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)  # 计算批次大小
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )  # 将窗口张量视为原始形状
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)  # 重新排列

    if Hp > H or Wp > W:  # 如果高度或宽度超过原始大小
        x = x[:, :H, :W, :].contiguous()  # 移除填充部分
    return x  # 返回恢复后的张量


class PatchEmbed(nn.Module):
    """
    图像到Patch的嵌入
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        参数:
            kernel_size (Tuple): 投影层的卷积核大小。
            stride (Tuple): 投影层的步幅。
            padding (Tuple): 投影层的填充大小。
            in_chans (int): 输入图像的通道数。
            embed_dim (int): Patch嵌入的维度。
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )  # 定义2D卷积层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # 应用卷积投影
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)  # 改变张量的维度顺序
        return x  # 返回嵌入后的张量
