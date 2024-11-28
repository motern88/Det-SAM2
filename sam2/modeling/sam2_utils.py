# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num, preloading_memory_cond_frame_idx=None):
    """
    从 `cond_frame_outputs` 中选择最多 `max_cond_frame_num` 个与当前帧 `frame_idx` 时间上最接近的条件帧。
    - a) 选择 `frame_idx` 之前的最近条件帧（如果有的话）；
    - b) 选择 `frame_idx` 之后的最近条件帧（如果有的话）；
    - c) 选择其他时间上最接近的条件帧，直到总数达到 `max_cond_frame_num`。

    输出：
    - selected_outputs: 从 `cond_frame_outputs` 中选择的项目（键和值）。
    - unselected_outputs: `cond_frame_outputs` 中未选择的项目（键和值）。
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        # 如果 `max_cond_frame_num` 为 -1 或条件帧的数量不超过最大条件帧数，直接返回所有输出
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "我们应该允许使用2个或更多条件帧"
        selected_outputs = {}

        # 选择 `frame_idx` 之前的最近条件帧（如果有的话）
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # 选择 `frame_idx` 之后的最近条件帧（如果有的话）
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # 选择其他时间上最接近的条件帧，直到总数达到 `max_cond_frame_num`
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)

        if preloading_memory_cond_frame_idx is not None:
            # 如果有预加载内存库中的条件帧，将其添加到选择的帧中
            for t in preloading_memory_cond_frame_idx:
                if t not in selected_outputs.keys():
                    selected_outputs[t] = cond_frame_outputs[t]  # 将预加载帧加入到已选帧中

        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    获取1D正弦位置嵌入，按照原始Transformer论文中的方法。
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """根据字符串返回激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation 应该是 relu/gelu，而不是 {activation}.")


def get_clones(module, N):
    """
    返回 `module` 的 N 个克隆
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    # 修改自 https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob  # dropout 概率
        self.scale_by_keep = scale_by_keep  # 是否按保留概率缩放

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            # 如果 dropout 概率为0或者不在训练模式下，直接返回输入
            return x
        keep_prob = 1 - self.drop_prob  # 计算保留概率
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 生成与输入相同的形状
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)  # 根据保留概率生成随机张量
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)  # 按保留概率进行缩放
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()  # 初始化激活函数

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)  # 如果需要，应用 sigmoid 激活函数
        return x


# 修改自 https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# 来源于 https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # 权重参数
        self.bias = nn.Parameter(torch.zeros(num_channels))  # 偏置参数
        self.eps = eps  # 平滑因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)  # 计算均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用权重和偏置
        return x

def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM 默认值
    noise_bound: int = 20,  # SAM 默认值
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    为给定的 `bbox` 采样加噪的左上角和右下角坐标

    输入:
    - masks: [B, 1, H, W] 形状的 box 掩码, 类型为 torch.Tensor
    - noise: 噪声值，表示为 box 宽度和高度的比例，类型为 float
    - noise_bound: 噪声的最大限制（以像素为单位），类型为 int

    输出:
    - box_coords: [B, num_pt, 2]，包含左上角和右下角 box 角点的 (x, y) 坐标，类型为 torch.float
    - box_labels: [B, num_pt]，label 2 表示左上角，3 表示右下角，类型为 torch.int32
    """
    device = masks.device
    box_coords = mask_to_box(masks)  # 将掩码转换为 box 坐标
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)

    # 加入噪声
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # 不使用中心化的像素坐标
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # 在坐标范围内进行裁剪

    box_coords = box_coords.reshape(-1, 2, 2)  # 始终返回 2 个点
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    从错误区域随机采样 `num_pt` 个点及其标签

    输入:
    - gt_masks: [B, 1, H_im, W_im]，ground truth 掩码，类型为 torch.bool
    - pred_masks: [B, 1, H_im, W_im]，预测掩码，类型为 torch.bool 或 None
    - num_pt: int，表示要独立采样的点数

    输出:
    - points: [B, num_pt, 2]，类型为 torch.float，包含每个采样点的 (x, y) 坐标
    - labels: [B, num_pt]，类型为 torch.int32，1 表示正点击，0 表示负点击
    """
    if pred_masks is None:  # 如果未提供 pred_masks，视为空
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive 区域：采样点应有负标签以修正 FP 错误
    fp_masks = ~gt_masks & pred_masks
    # false negative 区域：采样点应有正标签以修正 FN 错误
    fn_masks = gt_masks & ~pred_masks
    # 预测是否与 ground-truth 完全匹配
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # 通道 0 是 FP 区域图，通道 1 是 FN 区域图
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    从每个错误区域的中心采样 1 个点（错误区域到边界最远的点），使用 RITM 方法

    输入:
    - gt_masks: [B, 1, H_im, W_im] 掩码，类型为 torch.bool
    - pred_masks: [B, 1, H_im, W_im] 掩码，类型为 torch.bool 或 None
    - padding: 是否在边界处填充 1 像素，用于距离变换

    输出:
    - points: [B, 1, 2]，类型为 torch.float，包含每个采样点的 (x, y) 坐标
    - labels: [B, 1]，类型为 torch.int32，1 表示正点击，0 表示负点击
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive 区域，采样负标签
    fp_masks = ~gt_masks & pred_masks
    # false negative 区域，采样正标签
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels

def get_next_point(gt_masks, pred_masks, method):
    """
    根据指定的采样方法获取下一个点
    """
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"未知的采样方法 {method}")