# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model  # 模型的嵌入维度
        self.dim_feedforward = dim_feedforward  # 前馈层的维度
        self.dropout_value = dropout  # dropout比率
        self.self_attn = self_attention  # 自注意力层
        self.cross_attn_image = cross_attention  # 交叉注意力层

        # 前馈网络实现 Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 第一个线性层
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 第二个线性变换

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation  # 激活函数名称
        self.activation = get_activation_fn(activation)  # 激活函数

        # 位置编码添加位置的标志 Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn  # 是否在自注意力中添加位置编码
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries  # 是否在交叉注意力查询中添加位置编码
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys  # 是否在交叉注意力键中添加位置编码

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)  # 对目标进行归一化
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2  # 根据标志选择是否添加位置编码
        tgt2 = self.self_attn(q, k, v=tgt2)  # 自注意力计算
        tgt = tgt + self.dropout1(tgt2)  # 残差连接和dropout
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)  # 确保交叉注意力层是RoPEAttention类型
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}  # 传递额外参数

        # Cross-Attention
        tgt2 = self.norm2(tgt)  # 对目标进行归一化
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,  # 根据标志选择是否添加位置编码
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,  # 根据标志选择是否添加位置编码
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)  # 残差连接和dropout
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)  # 执行自注意力
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)  # 执行交叉注意力
        # MLP
        tgt2 = self.norm3(tgt)  # 对目标进行归一化
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))  # 通过前馈网络
        tgt = tgt + self.dropout3(tgt2)  # 残差连接和dropout
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # layer是否期望batch在前面
    ):
        super().__init__()
        self.d_model = d_model  # 模型的嵌入维度
        self.layers = get_clones(layer, num_layers)  # 复制层以构建多个层
        self.num_layers = num_layers  # 层数
        self.norm = nn.LayerNorm(d_model)  # 归一化层
        self.pos_enc_at_input = pos_enc_at_input  # 是否在输入时添加位置编码
        self.batch_first = batch_first  # 是否将batch维度放在前面

    def forward(
        self,
        curr: torch.Tensor,  # 自注意力输入 self-attention inputs , 张量torch.Size([4096, Batch, 256])
        memory: torch.Tensor,  # 交叉注意力输入 cross-attention inputs , 张量torch.Size([N, Batch, 256])
        curr_pos: Optional[Tensor] = None,  # 自注意力输入的位置编码 pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # 交叉注意力输入的位置编码 pos_enc for cross-attention inputs , 形状和memory一致
        num_obj_ptr_tokens: int = 0,  # 对象指针的token数量 number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)  # 确保位置编码也是列表
            assert len(curr) == len(curr_pos) == 1  # 确保列表长度为1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "curr和memory的batch大小必须相同"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos  # 在输入时添加位置编码

        if self.batch_first:
            # 将数据转换为batch在前的格式 Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        # 正常情况下两者均为torch.float32,
        # 报错时memory:torch.bfloat16 ,output:torch.float32
        assert (
            memory.dtype == output.dtype
        ), f"output和memory的数据类型必须相同，memory.dtype:{memory.dtype},output.dtype:{output.dtype}"


        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}  # 传递对象指针token的数量

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)  # 对最终输出进行归一化

        if self.batch_first:
            # 将数据转换回seq在前的格式 Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
