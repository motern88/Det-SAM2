# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        对输入提示进行编码，用于SAM的掩码解码器。

        参数:
          embed_dim (int): 提示的嵌入维度
          image_embedding_size (tuple(int, int)): 图像嵌入的空间大小，以(H, W)表示
          input_image_size (int): 输入图像到图像编码器的填充大小，以(H, W)表示
          mask_in_chans (int): 用于编码输入掩码的隐藏通道数
          activation (nn.Module): 编码输入掩码时使用的激活函数
        """
        super().__init__()
        self.embed_dim = embed_dim  # 嵌入维度
        self.input_image_size = input_image_size  # 输入图像大小
        self.image_embedding_size = image_embedding_size  # 图像嵌入大小
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)  # 位置编码层

        self.num_point_embeddings: int = 4  # 点嵌入的数量（正点/负点 + 2个框角点）
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)  # 点嵌入模块列表
        self.not_a_point_embed = nn.Embedding(1, embed_dim)  # 非点嵌入

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )  # 掩码输入的大小
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),  # 下采样卷积层
            LayerNorm2d(mask_in_chans // 4),  # 层归一化
            activation(),  # 激活函数
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),  # 第二层下采样卷积层
            LayerNorm2d(mask_in_chans),  # 层归一化
            activation(),  # 激活函数
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),  # 最后一层卷积层，将通道数转换为嵌入维度
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)  # 无掩码的嵌入

    def get_dense_pe(self) -> torch.Tensor:
        """
        返回用于编码点提示的位置信息编码，应用于与图像编码形状相同的密集点集。

        返回:
          torch.Tensor: 位置编码，形状为 1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 获取位置编码并增加一个维度

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """对点提示进行编码。"""
        points = points + 0.5  # 移动到像素中心
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)  # 填充点
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)  # 填充标签
            points = torch.cat([points, padding_point], dim=1)  # 拼接填充点
            labels = torch.cat([labels, padding_label], dim=1)  # 拼接填充标签
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )  # 使用坐标获取点嵌入
        point_embedding[labels == -1] = 0.0  # 非点的嵌入设置为0
        point_embedding[labels == -1] += self.not_a_point_embed.weight  # 非点嵌入加上对应权重
        point_embedding[labels == 0] += self.point_embeddings[0].weight  # 标签为0的点嵌入加上对应权重
        point_embedding[labels == 1] += self.point_embeddings[1].weight  # 标签为1的点嵌入加上对应权重
        point_embedding[labels == 2] += self.point_embeddings[2].weight  # 标签为2的点嵌入加上对应权重
        point_embedding[labels == 3] += self.point_embeddings[3].weight  # 标签为3的点嵌入加上对应权重
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """对框提示进行编码。"""
        boxes = boxes + 0.5  # 移动到像素中心
        coords = boxes.reshape(-1, 2, 2)  # 重塑框的坐标
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )  # 使用坐标获取角点嵌入
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # 第一角点加上对应权重
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # 第二角点加上对应权重
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """对掩码输入进行编码。"""
        mask_embedding = self.mask_downscaling(masks)  # 使用下采样模块编码掩码
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        根据输入提示的批次大小获取输出的批次大小。
        """
        if points is not None:
            return points[0].shape[0]  # 如果有点提示，返回点的批次大小
        elif boxes is not None:
            return boxes.shape[0]  # 如果有框提示，返回框的批次大小
        elif masks is not None:
            return masks.shape[0]  # 如果有掩码，返回掩码的批次大小
        else:
            return 1  # 如果没有任何提示，返回默认的批次大小1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device  # 获取点嵌入的设备

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对不同类型的提示进行编码，返回稀疏和密集的嵌入。

        参数:
          points (tuple(torch.Tensor, torch.Tensor) 或 None): 要编码的点坐标和标签
          boxes (torch.Tensor 或 None): 要编码的框
          masks (torch.Tensor 或 None): 要编码的掩码

        返回:
          torch.Tensor: 点和框的稀疏嵌入，形状为 BxNx(embed_dim)，N由输入的点和框数量决定
          torch.Tensor: 掩码的密集嵌入，形状为 Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)  # 获取批次大小
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )  # 初始化稀疏嵌入
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)  # 拼接点嵌入
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)  # 拼接框嵌入

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)  # 获取掩码的密集嵌入
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )  # 使用无掩码的嵌入填充

        return sparse_embeddings, dense_embeddings  # 返回稀疏嵌入和密集嵌入
