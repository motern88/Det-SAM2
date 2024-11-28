# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import numpy as np
import torch

# Very lightly adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py


class MaskData:
    """
    存储掩码及其相关数据的结构，采用批处理格式。
    实现基本的过滤和拼接功能。
    """

    def __init__(self, **kwargs) -> None:
        # 确保所有传入的参数是 list、numpy 数组或 torch 张量
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData 仅支持 list、numpy 数组和 torch 张量。"
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        # 确保设置的项是 list、numpy 数组或 torch 张量
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData 仅支持 list、numpy 数组和 torch 张量。"
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: torch.Tensor) -> None:
        """
        根据 `keep` 过滤数据，只保留为 True 的项。
        """
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData 键 {k} 的类型 {type(v)} 不受支持。")

    def cat(self, new_stats: "MaskData") -> None:
        """
        将新的数据拼接到现有数据中。
        """
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData 键 {k} 的类型 {type(v)} 不受支持。")

    def to_numpy(self) -> None:
        """
        将所有张量数据转换为 numpy 数组。
        """
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.float().detach().cpu().numpy()


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    """
    过滤接近裁剪边缘的掩码，但不包括原始图像边缘。
    """
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    # 检查是否接近裁剪边缘
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    # 检查是否接近原始图像边缘
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    # 只保留接近裁剪边缘但不接近原始图像边缘的框
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    """
    将边界框坐标从 (x1, y1, x2, y2) 转换为 (x, y, w, h) 格式。
    """
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]  # 计算宽度
    box_xywh[3] = box_xywh[3] - box_xywh[1]  # 计算高度
    return box_xywh


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """
    按指定的 batch_size 迭代数据。
    """
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "批处理迭代必须具有相同大小的输入。"
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    将掩码编码为无压缩的 RLE（游程编码），格式与 pycoco 工具兼容。
    """
    # 转置为 Fortran 顺序并展平 h, w 维度
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # 计算变化的索引
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # 编码游程长度
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """从无压缩的 RLE 计算二值掩码。"""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)  # 创建一个空的布尔数组
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity  # 根据游程长度填充掩码
        idx += count
        parity ^= True  # 切换掩码值（0 或 1）
    mask = mask.reshape(w, h)  # 重新调整形状为 (w, h)
    return mask.transpose()  # 转置为 C 顺序


def area_from_rle(rle: Dict[str, Any]) -> int:
    """从 RLE 计算区域面积。"""
    return sum(rle["counts"][1::2])  # 计算游程长度中奇数索引的位置和


def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    计算掩码的稳定性分数。稳定性分数是通过对预测掩码 logits 进行高低值阈值化，计算得到的 IoU。
    """
    # 一个掩码总是包含在另一个掩码内。
    # 通过防止不必要的转换到 torch.int64 来节省内存
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """生成 [0,1]x[0,1] 区间内均匀分布的 2D 点网格。"""
    offset = 1 / (2 * n_per_side)  # 网格点偏移量
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)  # 在 [0,1] 范围内生成点
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))  # 生成 x 方向的点网格
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))  # 生成 y 方向的点网格
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)  # 生成网格点并展平
    return points


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """为所有裁剪层生成点网格。"""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))  # 计算每层的点数
        points_by_layer.append(build_point_grid(n_points))  # 生成每层的点网格
    return points_by_layer


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    生成不同大小的裁剪框列表。每一层有 (2**i)**2 个框。
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)  # 短边长度

    # 原始图像
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """计算裁剪框的长度。"""
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)  # 每层每边的裁剪框数量
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))  # 计算重叠区域

        crop_w = crop_len(im_w, n_crops_per_side, overlap)  # 计算裁剪框宽度
        crop_h = crop_len(im_h, n_crops_per_side, overlap)  # 计算裁剪框高度

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]  # 计算 x0 坐标
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]  # 计算 y0 坐标

        # XYWH 格式的裁剪框
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """将裁剪框内的边界框坐标恢复到原始图像的坐标系中。"""
    x0, y0, _, _ = crop_box  # 裁剪框的左上角坐标
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)  # 计算偏移量
    # 检查 boxes 是否具有通道维度
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)  # 如果有通道维度，则扩展偏移量的维度
    return boxes + offset  # 恢复边界框坐标


def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """将裁剪框内的点坐标恢复到原始图像的坐标系中。"""
    x0, y0, _, _ = crop_box  # 裁剪框的左上角坐标
    offset = torch.tensor([[x0, y0]], device=points.device)  # 计算偏移量
    # 检查 points 是否具有通道维度
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)  # 如果有通道维度，则扩展偏移量的维度
    return points + offset  # 恢复点坐标


def uncrop_masks(
    masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int
) -> torch.Tensor:
    """将裁剪后的掩码恢复到原始图像的尺寸。"""
    x0, y0, x1, y1 = crop_box  # 裁剪框的坐标
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks  # 如果裁剪框与原始图像一致，直接返回原掩码
    # 坐标转换，恢复掩码到原始尺寸
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)  # 计算填充量
    pad = (x0, pad_x - x0, y0, pad_y - y0)  # 填充参数
    return torch.nn.functional.pad(masks, pad, value=0)  # 使用填充恢复掩码


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    移除掩码中小的孤立区域和孔洞。返回处理后的掩码以及掩码是否被修改的标志。
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]  # 验证模式是否正确
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)  # 将掩码转换为正确的工作模式
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)  # 获取连通区域
    sizes = stats[:, -1][1:]  # 获取区域的大小（排除背景标签）
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]  # 小区域的标签
    if len(small_regions) == 0:
        return mask, False  # 如果没有小区域，则不修改掩码
    fill_labels = [0] + small_regions  # 包含背景标签和小区域标签
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]  # 如果模式不是“holes”，则保留大区域
        # 如果所有区域都小于阈值，则保留最大的区域
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)  # 更新掩码
    return mask, True  # 返回更新后的掩码和修改标志


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    """将无压缩 RLE 编码转换为 COCO 格式。"""
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]  # 获取掩码的尺寸
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)  # 转换为 COCO 格式
    rle["counts"] = rle["counts"].decode("utf-8")  # 将 counts 转换为 UTF-8 格式，以便于 JSON 序列化
    return rle


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    计算掩码的 XYXY 格式边界框。对于空掩码，返回 [0,0,0,0]。输入形状为 C1xC2x...xHxW，输出形状为 C1xC2x...x4。
    """
    # torch.max 在空输入上会引发错误，遇到这种情况直接跳过
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # 规范化形状为 CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)  # 展平通道维度
    else:
        masks = masks.unsqueeze(0)  # 增加批量维度

    # 获取上下边界
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # 获取左右边界
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # 如果掩码为空，则右边界会在左边界的左侧。用 [0, 0, 0, 0] 替换这些边界框
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)  # 替换空掩码边界框为 [0, 0, 0, 0]

    # 返回到原始形状
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out
