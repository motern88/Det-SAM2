# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    这是一个标准版本的位置嵌入，非常类似于"Attention Is All You Need"论文中使用的版本，
    该版本被泛化以适用于图像。
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "期望模型宽度为偶数"  # 确保嵌入维度是偶数
        self.num_pos_feats = num_pos_feats // 2  # 位置嵌入的特征数的一半
        self.temperature = temperature  # 温度参数，用于控制位置编码的频率
        self.normalize = normalize  # 是否对位置进行归一化
        if scale is not None and normalize is False:
            raise ValueError("如果传递了scale，normalize应该为True")  # 如果scale被指定，normalize必须为True
        if scale is None:
            scale = 2 * math.pi  # 默认使用2*pi作为scale
        self.scale = scale

        self.cache = {}  # 缓存计算过的位置编码

    def _encode_xy(self, x, y):
        # 假定位置是归一化后的 The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1  # 确保x和y的长度及维度一致
        x_embed = x * self.scale  # 将x坐标放缩
        y_embed = y * self.scale  # 将y坐标放缩

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 计算频率基数

        pos_x = x_embed[:, None] / dim_t  # 计算x方向的位置编码
        pos_y = y_embed[:, None] / dim_t  # 计算y方向的位置编码
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)  # 使用sin和cos生成x方向的位置编码
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)  # 使用sin和cos生成y方向的位置编码
        return pos_x, pos_y  # 返回x和y方向的位置编码

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)  # 对x和y坐标进行编码
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)  # 拼接y、x编码以及高度和宽度
        return pos

    encode = encode_boxes  # 向后兼容，encode等价于encode_boxes。 Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape  # 获取x、y和标签的形状
        assert bx == by and nx == ny and bx == bl and nx == nl  # 确保x、y和标签的形状匹配
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())  # 展平后对x和y坐标进行编码
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)  # 重塑为原始形状
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)  # 拼接y、x编码和标签
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])  # 根据输入的空间大小生成缓存键
        if cache_key in self.cache:  # 如果缓存中已经有相应的编码
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)  # 返回缓存的编码
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])  # 生成y方向的嵌入
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)  # 生成x方向的嵌入
        )

        if self.normalize:  # 如果需要归一化
            eps = 1e-6  # 避免除以0
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 归一化y方向的嵌入
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 归一化x方向的嵌入

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 计算频率基数

        pos_x = x_embed[:, :, :, None] / dim_t  # 计算x方向的位置编码
        pos_y = y_embed[:, :, :, None] / dim_t  # 计算y方向的位置编码
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)  # 使用sin和cos进行交替堆叠并展平，生成x方向的位置编码
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)  # 使用sin和cos进行交替堆叠并展平，生成y方向的位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 拼接并调整维度顺序
        self.cache[cache_key] = pos[0]  # 缓存位置编码
        return pos  # 返回位置编码


class PositionEmbeddingRandom(nn.Module):
    """
    使用随机空间频率的坐标编码。
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0  # 如果没有传递scale或者scale为非正值，默认scale为1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),  # 使用高斯随机矩阵作为位置编码基矩阵
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """对已归一化到[0,1]的点进行位置编码。"""
        # 假设 coords 在 [0, 1]^2 正方形内且为 d_1 x ... x d_n x 2 形状
        coords = 2 * coords - 1  # 将坐标从[0, 1]转换到[-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix  # 通过基矩阵进行线性变换
        coords = 2 * np.pi * coords  # 乘以2π进行周期转换
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # 返回正弦和余弦编码的拼接结果

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """为指定大小的网格生成位置编码。"""
        h, w = size  # 获取网格的高度和宽度
        device: Any = self.positional_encoding_gaussian_matrix.device  # 获取设备信息
        grid = torch.ones((h, w), device=device, dtype=torch.float32)  # 创建一个全1的网格
        y_embed = grid.cumsum(dim=0) - 0.5  # 计算y方向的累加和并减去0.5
        x_embed = grid.cumsum(dim=1) - 0.5  # 计算x方向的累加和并减去0.5
        y_embed = y_embed / h  # 归一化y方向嵌入
        x_embed = x_embed / w  # 归一化x方向嵌入

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # 计算位置编码
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """对未归一化到[0,1]范围内的点进行位置编码。"""
        coords = coords_input.clone()  # 克隆输入的坐标张量
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # 将 x 坐标除以图像宽度进行归一化
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # 将 y 坐标除以图像高度进行归一化
        return self._pe_encoding(coords.to(torch.float))  # 对归一化后的坐标进行位置编码，返回 B x N x C 的张量


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)  # 创建长度为 end_x * end_y 的一维张量
    t_x = (t % end_x).float()  # 计算 t 对 end_x 的取余，得到 x 方向上的位置
    t_y = torch.div(t, end_x, rounding_mode="floor").float()  # 计算 t 除以 end_x 的商，得到 y 方向上的位置
    return t_x, t_y  # 返回 x 和 y 方向上的位置

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))  # 计算 x 方向上的频率
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))  # 计算 y 方向上的频率

    t_x, t_y = init_t_xy(end_x, end_y)  # 初始化 x 和 y 方向上的位置
    freqs_x = torch.outer(t_x, freqs_x)  # 计算 x 方向上的外积得到频率张量
    freqs_y = torch.outer(t_y, freqs_y)  # 计算 y 方向上的外积得到频率张量
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)  # 生成 x 方向的复数极坐标表示
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)  # 生成 y 方向的复数极坐标表示
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)  # 将 x 和 y 方向的极坐标频率拼接在一起


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim  # 获取输入张量的维度
    assert 0 <= 1 < ndim  # 确保输入张量至少有两个维度
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])  # 确保频率张量的形状与输入张量的最后两个维度匹配
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]  # 调整频率张量的形状以便广播
    return freqs_cis.view(*shape)  # 返回调整后的频率张量


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 将查询向量转换为复数表示
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )  # 如果键向量非空，将其转换为复数表示，否则设为 None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # 调整频率张量以便与查询向量进行广播
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # 将查询向量进行旋转编码后转换回实数表示
    if xk_ is None:
        # 没有键向量需要旋转编码，可能是由于 dropout
        return xq_out.type_as(xq).to(xq.device), xk  # 返回处理后的查询向量和原始键向量
    # 沿序列长度维度重复频率张量以匹配键向量的序列长度
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]  # 计算键向量序列长度与查询向量序列长度的比率
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)  # 在 GPU 上重复频率张量
        else:
            # torch.repeat 可能不支持在非 CUDA 设备上对复数进行操作
            # （freqs_cis 具有 4 维，且我们在第 2 维度上进行重复），所以我们使用 expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # 将键向量进行旋转编码后转换回实数表示
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)  # 返回处理后的查询向量和键向量
