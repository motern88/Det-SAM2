# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,  # 主干网络
        neck: nn.Module,  # 颈部网络
        scalp: int = 0,  # 头皮层数
    ):
        super().__init__()  # 初始化
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        # 验证主干网络和颈部网络的通道数是否匹配
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # 通过主干网络进行前向传播
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # 如果头皮层数大于0，丢弃分辨率最低的特征
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]  # 获取最高分辨率的特征
        output = {
            "vision_features": src,  # 视觉特征
            "vision_pos_enc": pos,  # 视觉位置编码
            "backbone_fpn": features,  # 主干网络的特征金字塔网络(FPN)
        }
        return output


# 定义特征金字塔网络(FPN)颈部类
class FpnNeck(nn.Module):
    """
    修改后的特征金字塔网络(FPN)颈部类
    （我们移除了输出卷积层，并且采用类似ViT的位置嵌入插值的双三次插值方法）
    """

    def __init__(
        self,
        position_encoding: nn.Module,  # 位置编码模块
        d_model: int,  # 模型维度
        backbone_channel_list: List[int],  # 主干网络的通道列表
        kernel_size: int = 1,  # 卷积核大小
        stride: int = 1,  # 步幅
        padding: int = 0,  # 填充
        fpn_interp_model: str = "bilinear",  # FPN插值模型
        fuse_type: str = "sum",  # 特征融合类型
        fpn_top_down_levels: Optional[List[int]] = None,  # FPN自顶向下的层级
    ):
        """初始化FPN颈部模块
        :param trunk: 主干网络
        :param position_encoding: 使用的位置编码模块
        :param d_model: 模型维度
        :param neck_norm: 使用的归一化方法
        """
        super().__init__()
        self.position_encoding = position_encoding  # 初始化位置编码
        self.convs = nn.ModuleList()  # 创建卷积模块列表
        self.backbone_channel_list = backbone_channel_list  # 初始化主干网络通道列表
        for dim in backbone_channel_list:
            current = nn.Sequential()  # 为每个通道创建一个顺序容器
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,  # 输入通道数
                    out_channels=d_model,  # 输出通道数
                    kernel_size=kernel_size,  # 卷积核大小
                    stride=stride,  # 步幅
                    padding=padding,  # 填充
                ),
            )

            self.convs.append(current)  # 将当前卷积模块添加到模块列表
        self.fpn_interp_model = fpn_interp_model  # 初始化插值模型
        assert fuse_type in ["sum", "avg"]  # 确认融合类型为"sum"或"avg"
        self.fuse_type = fuse_type  # 初始化融合类型

        # 确定输出中包含自顶向下特征的层级
        # 例如，如果fpn_top_down_levels是[2, 3]，那么只有第2和第3层级包含自顶向下的传播，
        # 而第0和第1层级则仅包含来自相同主干网络层级的横向特征。
        if fpn_top_down_levels is None:
            # 默认情况下，所有层级都包含自顶向下特征
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)  # 初始化自顶向下的层级列表

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)  # 初始化输出列表
        pos = [None] * len(self.convs)  # 初始化位置编码列表
        assert len(xs) == len(self.convs)  # 验证输入的张量数是否与卷积模块数匹配
        # FPN前向传播
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None  # 初始化前一层的特征
        # 以自顶向下的顺序（从低分辨率到高分辨率）进行前向传播
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)  # 计算横向特征
            if i in self.fpn_top_down_levels and prev_features is not None:
                # 如果当前层级在自顶向下的层级列表中且存在前一层特征，计算自顶向下特征
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),  # 对前一层特征进行插值
                    scale_factor=2.0,  # 放大2倍
                    mode=self.fpn_interp_model,  # 使用指定的插值模式
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),  # 对齐角落（如果插值模式为"nearest"，则不对齐）
                    antialias=False,  # 不进行抗锯齿处理
                )
                prev_features = lateral_features + top_down_features  # 横向特征与自顶向下特征相加
                if self.fuse_type == "avg":
                    prev_features /= 2  # 如果融合类型为"avg"，则特征求平均
            else:
                prev_features = lateral_features  # 否则，仅使用横向特征
            x_out = prev_features  # 设置当前输出
            out[i] = x_out  # 将输出保存到输出列表中
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)  # 计算并保存位置编码

        return out, pos  # 返回输出特征和位置编码
