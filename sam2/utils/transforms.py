# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        SAM2 的Transforms。
        """
        super().__init__()
        self.resolution = resolution  # 目标分辨率
        self.mask_threshold = mask_threshold  # 掩码阈值
        self.max_hole_area = max_hole_area  # 最大孔洞面积
        self.max_sprinkle_area = max_sprinkle_area  # 最大撒布面积
        self.mean = [0.485, 0.456, 0.406]  # 图像均值
        self.std = [0.229, 0.224, 0.225]  # 图像标准差
        self.to_tensor = ToTensor()  # 转换为张量的操作
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),  # 调整图像大小
                Normalize(self.mean, self.std),  # 标准化图像
            )
        )

    def __call__(self, x):
        """
        将输入图像 `x` 转换为张量并应用预定义的变换操作。
        """
        x = self.to_tensor(x)  # 将图像转换为张量
        return self.transforms(x)  # 应用图像变换

    def forward_batch(self, img_list):
        """
        对图像列表 `img_list` 应用变换操作，并返回一个图像批次。
        """
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]  # 对每张图像进行变换
        img_batch = torch.stack(img_batch, dim=0)  # 将图像列表堆叠成一个批次
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        将坐标 `coords` 转换为 SAM2 模型期望的坐标格式。
        坐标可以是绝对图像坐标或归一化坐标。如果坐标是绝对图像坐标，
        则应将 `normalize` 设置为 True，并提供原始图像大小。

        返回：
            转换后的坐标，范围为 [0, 1]，这是 SAM2 模型所期望的。
        """
        if normalize:
            assert orig_hw is not None  # 确保提供了原始图像大小
            h, w = orig_hw  # 获取原始图像的高度和宽度
            coords = coords.clone()  # 克隆坐标以避免修改原始数据
            coords[..., 0] = coords[..., 0] / w  # 将 x 坐标归一化
            coords[..., 1] = coords[..., 1] / h  # 将 y 坐标归一化

        coords = coords * self.resolution  # 将坐标反归一化到指定分辨率
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        将边界框 `boxes` 转换为 SAM2 模型所需的格式。
        边界框的形状应为 Bx4。坐标可以是绝对图像坐标或归一化坐标，
        如果坐标是绝对图像坐标，则应将 `normalize` 设置为 True，并提供原始图像大小。
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)  # 转换边界框坐标
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        对输出掩码进行后处理。
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()  # 将掩码转换为浮点类型
        input_masks = masks  # 保存原始掩码以防后续处理失败
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # 将掩码展平为单通道图像
        try:
            if self.max_hole_area > 0:
                # 将背景区域中面积 <= self.max_hole_area 的连接组件视为孔洞
                # 背景区域的掩码分数 <= self.mask_threshold
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)  # 判断哪些是孔洞
                is_hole = is_hole.reshape_as(masks)  # 将孔洞掩码调整为原始形状
                # 用小的正掩码分数 (10.0) 填充孔洞，将其变为前景
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                # 将前景区域中面积 <= self.max_sprinkle_area 的连接组件视为撒布
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)  # 判断哪些是撒布
                is_hole = is_hole.reshape_as(masks)  # 将撒布掩码调整为原始形状
                # 用负掩码分数 (-10.0) 填充撒布区域，将其变为背景
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # 如果 CUDA 内核失败，则跳过后处理步骤
            warnings.warn(
                f"{e}\n\n由于上述错误，跳过后处理步骤。您仍然可以使用 SAM 2，"
                "忽略上述错误通常是可以的，尽管某些后处理功能可能受限（这不会影响大多数情况下的结果；请参阅 "
                "https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md）。",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks  # 如果后处理失败，使用原始掩码

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)  # 调整掩码到原始图像大小
        return masks
