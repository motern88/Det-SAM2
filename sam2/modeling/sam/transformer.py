# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import get_sdpa_settings

warnings.simplefilter(action="ignore", category=FutureWarning)
# 检查是否可以使用 Flash Attention（默认使用），如果不行则使用所有可用的内核
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
# 如果 Flash Attention 失败，则允许使用所有可用的内核
ALLOW_ALL_KERNELS = False


def sdp_kernel_context(dropout_p):
    """
    获取注意力缩放点积内核的上下文。默认使用 Flash Attention，
    如果 Flash Attention 失败，则回退到所有可用的内核。
    """
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        # 如果 Flash Attention 内核关闭，则需要启用数学内核
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,
    )


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        一个 Transformer 解码器，通过查询来关注输入图像，查询的
        位置嵌入是提供的。

        参数：
          depth (int): Transformer 的层数
          embedding_dim (int): 输入嵌入的通道维度
          num_heads (int): 多头注意力的头数，必须能整除 embedding_dim
          mlp_dim (int): MLP 块内部的通道维度
          activation (nn.Module): 在 MLP 块中使用的激活函数
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        # 最终的注意力层，将点从图像中进行注意
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        参数：
          image_embedding (torch.Tensor): 要关注的图像，形状应为
            B x embedding_dim x h x w
          image_pe (torch.Tensor): 添加到图像上的位置编码，形状应与 image_embedding 相同
          point_embedding (torch.Tensor): 添加到查询点上的嵌入，形状应为
            B x N_points x embedding_dim

        返回：
          torch.Tensor: 处理后的 point_embedding
          torch.Tensor: 处理后的 image_embedding
        """
        # BxCxHxW -> BxHWxC 转换为 B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # 准备查询 Prepare queries
        queries = point_embedding
        keys = image_embedding

        # 应用 transformer 层和最终的 layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 应用最终的注意力层，将点从图像中进行注意
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        一个 Transformer 块，包含四层： (1) 稀疏输入的自注意力， (2) 稀疏输入对密集输入的交叉注意力，
        (3) 在稀疏输入上进行的 MLP 块，和 (4) 密集输入对稀疏输入的交叉注意力。

        参数：
          embedding_dim (int): 嵌入的通道维度
          num_heads (int): 注意力层中的头数
          mlp_dim (int): MLP 块中的隐藏维度
          activation (nn.Module): MLP 块中的激活函数
          skip_first_layer_pe (bool): 是否在第一层跳过位置编码
        """
        super().__init__()  # 初始化父类
        self.self_attn = Attention(embedding_dim, num_heads)  # 初始化自注意力机制（Self-Attention）
        self.norm1 = nn.LayerNorm(embedding_dim)  # 自注意力机制后的归一化层

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )  # 初始化token到图像的交叉注意力机制（Cross-Attention）
        self.norm2 = nn.LayerNorm(embedding_dim) # 交叉注意力机制后的归一化层

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )  # 初始化多层感知机（MLP）模块
        self.norm3 = nn.LayerNorm(embedding_dim)  # MLP后的归一化层

        self.norm4 = nn.LayerNorm(embedding_dim)  # 交叉注意力机制后的归一化层（图像到token）
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )  # 初始化图像到令牌的交叉注意力机制（Cross-Attention）

        self.skip_first_layer_pe = skip_first_layer_pe  # 是否跳过第一个层的位置信息编码（PE）

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # 自注意力块 Self attention block
        if self.skip_first_layer_pe:
            # 如果跳过第一个层的位置信息编码，则直接应用自注意力机制
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            # 否则，先将位置信息编码添加到查询（queries）中
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)  # 应用自注意力机制
            queries = queries + attn_out  # 将自注意力输出添加到查询中
        queries = self.norm1(queries)  # 对查询进行归一化

        # 交叉注意力块，token关注图像嵌入 Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)  # 应用令牌到图像的交叉注意力机制
        queries = queries + attn_out  # 将交叉注意力输出添加到查询中
        queries = self.norm2(queries)  # 对查询进行归一化

        # MLP block
        mlp_out = self.mlp(queries)  # 对查询应用MLP
        queries = queries + mlp_out  # 将MLP输出添加到查询中
        queries = self.norm3(queries)  # 对查询进行归一化

        # 交叉注意力块，图像嵌入关注token Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)  # 应用图像到令牌的交叉注意力机制
        keys = keys + attn_out  # 将交叉注意力输出添加到键中
        keys = self.norm4(keys)  # 对键进行归一化

        return queries, keys


class Attention(nn.Module):
    """
   一个注意力层，允许在对Q、K和V进行投影后，缩小嵌入的大小。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim  # 嵌入的维度
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim  # 键值对的输入维度
        self.internal_dim = embedding_dim // downsample_rate  # 内部维度，考虑降采样率
        self.num_heads = num_heads  # 注意力头的数量
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."  # 确保内部维度可以被头数整除

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape  # b: 批量大小，n: Token数量，c: 嵌入维度
        x = x.reshape(b, n, num_heads, c // num_heads)  # 将嵌入维度划分到多个头上
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape  # b: 批量大小，n_heads: 头的数量，n_tokens: 令牌数量，c_per_head: 每个头的特征维度
        x = x.transpose(1, 2)   # 交换head和token的维度
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 输入投影 Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 分离头 Separate into heads
        q = self._separate_heads(q, self.num_heads)  # 将查询分离到多个头上
        k = self._separate_heads(k, self.num_heads)  # 将键分离到多个头上
        v = self._separate_heads(v, self.num_heads)  # 将值分离到多个头上

        dropout_p = self.dropout_p if self.training else 0.0    # 如果是训练模式，则使用dropout概率
        # Attention计算
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)  # 计算缩放点积注意力
        except Exception as e:
            # 如果Flash Attention内核失败，则回退到所有内核
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True  # 允许使用所有可用内核
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)  # 重新计算缩放点积注意力

        out = self._recombine_heads(out)  # 重新组合头
        out = self.out_proj(out)  # 进行最终的线性投影

        return out


class RoPEAttention(Attention):
    """带有旋转位置编码的注意力层."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,  # 旋转位置编码的参数
        # 是否重复q的rope以匹配k的长度
        # 这对于记忆的交叉注意力是必要的
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] 表示512分辨率下的stride 16特征图尺寸
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # 调用父类构造函数

        # 初始化旋转位置编码的计算方法
        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        # 计算旋转位置编码的频率
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis  # 保存频率编码
        self.rope_k_repeat = rope_k_repeat  # 是否重复k的rope

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # 输入投影 Input projections
        q = self.q_proj(q)  # 对查询进行线性投影
        k = self.k_proj(k)  # 对键进行线性投影
        v = self.v_proj(v)  # 对值进行线性投影

        # 分离头 Separate into heads
        q = self._separate_heads(q, self.num_heads)  # 将查询分离到多个头上
        k = self._separate_heads(k, self.num_heads)  # 将键分离到多个头上
        v = self._separate_heads(v, self.num_heads)  # 将值分离到多个头上

        # 应用旋转位置编码 Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])  # 计算特征图的宽高
        self.freqs_cis = self.freqs_cis.to(q.device)  # 将频率编码移到当前设备
        if self.freqs_cis.shape[0] != q.shape[-2]:
            # 如果频率编码的大小与当前的查询维度不匹配，则重新计算频率编码
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            # 如果查询和键的长度不一致，且需要重复k的rope
            assert self.rope_k_repeat

        # 计算k的长度并进行旋转位置编码
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0  # 如果是训练模式，则使用dropout概率
        # 注意力计算 Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)  # 计算缩放点积注意力
        except Exception as e:
            # # 如果Flash Attention内核失败，则回退到所有内核
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True  # 允许使用所有可用内核
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)  # 重新计算缩放点积注意力

        out = self._recombine_heads(out)  # 重新组合头
        out = self.out_proj(out)  # 进行最终的线性投影

        return out
