# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从自定义的模块导入必要的工具函数和模块
from sam2.modeling.backbones.utils import (
    PatchEmbed,  # 用于将图像块嵌入成特征
    window_partition,  # 窗口划分函数
    window_unpartition,  # 窗口还原函数
)

from sam2.modeling.sam2_utils import DropPath, MLP  # 导入自定义的路径丢弃和多层感知器模块


# 定义一个池化操作函数
def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x  # 如果池化层为空，直接返回输入
    # (B, H, W, C) -> (B, C, H, W) 将张量的通道维度从最后一维转到第二维
    x = x.permute(0, 3, 1, 2)
    x = pool(x)  # 执行池化操作
    # (B, C, H', W') -> (B, H', W', C) 将张量的通道维度还原
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)  # 如果提供了归一化层，对张量进行归一化

    return x  # 返回处理后的张量


# 定义一个多尺度注意力模块
class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim # 定义一个多尺度注意力模块
        self.dim_out = dim_out  # 输出的维度
        self.num_heads = num_heads  # 注意力头的数量
        self.q_pool = q_pool  # 查询的池化层
        self.qkv = nn.Linear(dim, dim_out * 3)  # 线性层，生成查询、键、值
        self.proj = nn.Linear(dim_out, dim_out)  # 输出投影层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape  # 获取批量大小、高度、宽度、通道数
        # 生成q,k,v张量，并重塑为(B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # 将qkv拆分，形状为(B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # 如果有池化层，则对 q 进行池化 (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # 更新下采样后的高度和宽度
            q = q.reshape(B, H * W, self.num_heads, -1)

        # 使用torch的缩放点积注意力函数进行计算 Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # 将结果转置回原来的形状
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)  # 通过投影层进行线性变换

        return x  # 返回注意力模块的输出


# 定义一个多尺度的块结构
class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)  # 如果传入的是字符串，创建相应的归一化层

        self.dim = dim  # 输入维度
        self.dim_out = dim_out  # 输出维度
        self.norm1 = norm_layer(dim)  # 归一化层

        self.window_size = window_size  # 窗口大小

        self.pool, self.q_stride = None, q_stride  # 池化层和步幅
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )  # 如果有步幅，使用最大池化

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )  # 创建多尺度注意力模块
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # 丢弃路径

        self.norm2 = norm_layer(dim_out)  # 第二个归一化层
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)  # 如果输入和输出维度不同，添加线性投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # 保留输入作为捷径连接 B, H, W, C
        x = self.norm1(x)  # 进行归一化

        # 通过投影层进行池化，作为捷径连接的一部分 Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # 执行窗口划分 Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # 执行多尺度注意力模块 Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # 如果有步幅，更新窗口大小和形状 Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # 反向还原窗口划分
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)  # 添加捷径连接
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过MLP模块
        return x  # 返回块的输出


# 定义主结构，基于Hiera架构
class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # 初始嵌入维度
        num_heads: int = 1,  # 初始注意力头数
        drop_path_rate: float = 0.0,  # 随机深度丢弃率
        q_pool: int = 3,  # q_pool阶段数
        q_stride: Tuple[int, int] = (2, 2),  # 阶段之间的下采样步幅
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # 每个阶段的块数
        dim_mul: float = 2.0,  # 阶段切换时的维度乘数
        head_mul: float = 2.0,  # 阶段切换时的头数乘数
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),  # 窗口位置嵌入背景的空间尺寸
        # 每个阶段的窗口大小，当不使用全局注意力时
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),  # 每个阶段的窗口大小
        # 全局注意力在这些块中
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ), # 使用全局注意力的块
        return_interm_layers=True,  # 返回每个阶段的中间层特征
    ):
        super().__init__()

        assert len(stages) == len(window_spec)  # 确保stages和window_spec的长度相同
        self.window_spec = window_spec  # 保存窗口规格

        depth = sum(stages)  # 总的深度为所有阶段的块数量之和
        self.q_stride = q_stride  # 保存q池化的步幅
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]  # 计算每个阶段的结束块索引
        assert 0 <= q_pool <= len(self.stage_ends[:-1])  # 确保q_pool在合法范围内
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]  # 确定哪些块需要进行q池化
        self.return_interm_layers = return_interm_layers  # 是否返回中间层输出的标志

        # 定义补丁嵌入层
        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # 确定哪些块使用全局注意力
        self.global_att_blocks = global_att_blocks

        # 窗口位置嵌入参数 Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]    # 随机深度丢弃率的线性增长 stochastic depth decay rule

        cur_stage = 1  # 当前阶段初始化为1
        self.blocks = nn.ModuleList()  # 初始化块的列表

        for i in range(depth):
            dim_out = embed_dim  # 默认输出维度与输入维度相同
            # 与当前块滞后一个块，因此下一个阶段的第一个块
            # 使用上一个阶段的初始窗口大小和当前阶段的最终窗口大小
            window_size = self.window_spec[cur_stage - 1]  # 当前阶段的窗口大小

            if self.global_att_blocks is not None:  # 如果是全局注意力块，窗口大小设为0
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)  # 每个阶段开始时，更新输出维度
                num_heads = int(num_heads * head_mul)  # 更新注意力头的数量
                cur_stage += 1  # 切换到下一个阶段

            # 创建多尺度块
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],  # 当前块的路径丢弃率
                q_stride=self.q_stride if i in self.q_pool_blocks else None,  # 是否在此块进行q池化
                window_size=window_size,  # 窗口大小
            )

            embed_dim = dim_out  # 更新嵌入维度为当前块的输出维度
            self.blocks.append(block)  # 将块添加到块列表中

        # 确定返回的通道列表
        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    # 获取位置嵌入
    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw  # 提取高度和宽度
        window_embed = self.pos_embed_window  # 提取窗口位置嵌入
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")  # 对位置嵌入进行插值以匹配输入尺寸
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )  # 将窗口位置嵌入加到位置嵌入上
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # 调整维度顺序
        return pos_embed  # 返回位置嵌入

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # 添加位置嵌入
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []  # 初始化输出列表
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # 通过每个块处理x
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)  # 调整x的维度顺序
                outputs.append(feats)  # 将特征添加到输出列表中

        return outputs  # 返回所有输出

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)