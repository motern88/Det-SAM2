# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import (
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SAM2AutomaticMaskGenerator:
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        multimask_output: bool = True,
        **kwargs,
    ) -> None:
        """
        使用 SAM 2 模型为整个图像生成掩膜。
        在图像上生成点提示的网格，然后过滤低质量和重复的掩膜。
        默认设置为 SAM 2 和 HieraL 骨干网络。

        参数：
          model (Sam): 用于掩膜预测的 SAM 2 模型。
          points_per_side (int 或 None): 在图像的一侧采样的点数。总点数为 points_per_side**2。如果为 None，则 'point_grids' 必须提供显式的点采样。
          points_per_batch (int): 设置模型同时处理的点数。较高的数值可能更快，但使用更多的 GPU 内存。
          pred_iou_thresh (float): 过滤阈值 [0,1]，使用模型预测的掩膜质量。
          stability_score_thresh (float): 过滤阈值 [0,1]，使用掩膜在二值化模型预测时的稳定性。
          stability_score_offset (float): 计算稳定性评分时偏移的量。
          mask_threshold (float): 二值化掩膜 logits 的阈值。
          box_nms_thresh (float): 非极大值抑制中用于过滤重复掩膜的框 IoU 截止值。
          crop_n_layers (int): 如果 >0，则对图像的裁剪运行掩膜预测。设置要运行的层数，每层有 2**i_layer 个图像裁剪。
          crop_nms_thresh (float): 非极大值抑制中用于过滤不同裁剪之间重复掩膜的框 IoU 截止值。
          crop_overlap_ratio (float): 设置裁剪的重叠程度。在第一层裁剪中，裁剪将按图像长度的此比例重叠。后续层有更多裁剪时缩小此重叠。
          crop_n_points_downscale_factor (int): 层 n 中采样的每侧点数按 crop_n_points_downscale_factor**n 缩小。
          point_grids (list(np.ndarray) 或 None): 显式的点网格列表用于采样，标准化到 [0,1]。列表中的第 n 个网格用于第 n 层裁剪。与 points_per_side 排斥。
          min_mask_region_area (int): 如果 >0，将应用后处理来移除小于 min_mask_region_area 的掩膜中断开的区域和孔。需要 opencv。
          output_mode (str): 掩膜的返回形式。可以是 'binary_mask'、'uncompressed_rle' 或 'coco_rle'。'coco_rle' 需要 pycocotools。对于大分辨率，'binary_mask' 可能消耗大量内存。
          use_m2m (bool): 是否使用先前掩膜预测进行一步修正。
          multimask_output (bool): 是否在网格的每个点处输出多掩膜。
        """

        # 确保 'points_per_side' 和 'point_grids' 中正好有一个被提供
        assert (points_per_side is None) != (
            point_grids is None
        ), "必须提供 points_per_side 或 point_grid 中的一个。"

        # 如果提供了 'points_per_side'，则生成所有层的点网格
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        # 如果提供了 'point_grids'，则直接使用提供的点网格
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("不能同时为 points_per_side 和 point_grid 设为 None。")

        # 确保 'output_mode' 是已知的
        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"未知的 output_mode {output_mode}."
        # 如果 'output_mode' 是 'coco_rle'，则尝试导入 pycocotools
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        # 初始化 SAM2 图像预测器
        self.predictor = SAM2ImagePredictor(
            model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )
        # 设置其他参数
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        从 Hugging Face hub 加载预训练模型。

        参数：
          model_id (str): Hugging Face 仓库 ID。
          **kwargs: 传递给模型构造函数的附加参数。

        返回：
          (SAM2AutomaticMaskGenerator): 加载的模型实例。
        """
        from sam2.build_sam import build_sam2_hf

        # 使用 Hugging Face hub 加载 SAM2 模型
        sam_model = build_sam2_hf(model_id, **kwargs)
        # 返回一个 SAM2AutomaticMaskGenerator 实例
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        为给定的图像生成掩膜。

        参数：
          image (np.ndarray): 要生成掩膜的图像，格式为 HWC uint8。

        返回：
           list(dict(str, any)): 掩膜记录的列表。每个记录是一个字典，包含以下键：
             segmentation (dict(str, any) 或 np.ndarray): 掩膜。如果 output_mode='binary_mask'，则为 HW 形状的数组。否则，为包含 RLE 的字典。
             bbox (list(float)): 掩膜周围的框，格式为 XYWH。
             area (int): 掩膜的像素面积。
             predicted_iou (float): 模型对掩膜质量的预测。这会根据 pred_iou_thresh 参数进行过滤。
             point_coords (list(list(float))): 用于生成此掩膜的点坐标输入。
             stability_score (float): 掩膜质量的度量。这会根据 stability_score_thresh 参数进行过滤。
             crop_box (list(float)): 用于生成掩膜的图像裁剪，格式为 XYWH。
        """

        # 生成掩码
        mask_data = self._generate_masks(image)

        # 编码掩码
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # 写入掩码记录
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        """
        生成掩膜的内部方法，处理图像裁剪和掩膜生成。

        参数：
          image (np.ndarray): 要处理的图像，格式为 HWC uint8。

        返回：
          MaskData: 包含掩膜数据的对象。
        """
        orig_size = image.shape[:2]  # 获取原始图像的大小
        # 生成裁剪框和层索引
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # 迭代处理图像裁剪
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)  # 合并当前裁剪的数据

        # 移除裁剪之间的重复掩码
        if len(crop_boxes) > 1:
            # 优先考虑来自较小裁剪的掩膜
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # 类别
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)  # 根据非极大值抑制结果过滤数据
        data.to_numpy()  # 转换为 numpy 格式
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        """
        处理图像的裁剪区域，生成掩膜数据。

        参数：
          image (np.ndarray): 要处理的图像，格式为 HWC uint8。
          crop_box (List[int]): 裁剪框，格式为 [x0, y0, x1, y1]。
          crop_layer_idx (int): 裁剪层的索引。
          orig_size (Tuple[int, ...]): 原始图像的大小。

        返回：
          MaskData: 包含掩膜数据的对象。
        """
        # 解包裁剪框坐标
        x0, y0, x1, y1 = crop_box
        # 从图像中裁剪出指定区域
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        # 设置裁剪后的图像
        self.predictor.set_image(cropped_im)

        # 获取当前裁剪区域的点
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # 批量生成掩膜
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            data.cat(batch_data)  # 合并当前批次的数据
            del batch_data
        self.predictor.reset_predictor()

        # 移除裁剪区域内的重复掩码
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # 类别
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # 将掩膜框、点和裁剪框还原到原始图像坐标
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        """
        处理点的批次，生成掩膜数据。

        参数：
          points (np.ndarray): 点坐标数组。
          im_size (Tuple[int, ...]): 裁剪区域的大小。
          crop_box (List[int]): 裁剪框，格式为 [x0, y0, x1, y1]。
          orig_size (Tuple[int, ...]): 原始图像的大小。
          normalize (bool): 是否对点进行归一化处理。

        返回：
          MaskData: 包含掩膜数据的对象。
        """
        orig_h, orig_w = orig_size

        # 在此批次上运行模型
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        # 转换点坐标
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        # 生成掩膜和 IoU 预测
        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        # 序列化预测结果并存储在 MaskData 中
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks

        if not self.use_m2m:
            # 通过预测的 IoU 进行过滤
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # 计算并按稳定性得分进行过滤
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            # 使用先前的掩膜预测进行一步精化
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            # 通过预测的 IoU 进行过滤
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # 计算并按稳定性得分进行过滤
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # 对掩膜进行阈值处理，并计算掩码框
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # 过滤触及裁剪边界的掩码框
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # 压缩掩码为 RLE 格式
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        移除掩码中的小型断裂区域和孔洞，然后重新运行框 NMS 以去除任何新产生的重复项。

        直接修改 mask_data。

        需要安装 open-cv 作为依赖。
        """
        if len(mask_data["rles"]) == 0:
            return mask_data  # 如果没有掩膜，直接返回

        # 过滤小型断裂区域和孔洞
        new_masks = []  # 用于存储处理后的掩膜
        scores = []     # 用于存储每个掩膜的分数
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)  # 将 RLE 解码为掩膜

            # 去除小型孔洞和小型岛屿
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))  # 将掩码转为张量并添加到列表
            # 如果掩码未被改变，则分数为 1，否则为 0，以便 NMS 更倾向于未处理的掩码
            scores.append(float(unchanged))

        # 重新计算框并移除任何新产生的重复项
        masks = torch.cat(new_masks, dim=0)  # 合并所有掩码
        boxes = batched_mask_to_box(masks)  # 计算掩码的边界框
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # 类别
            iou_threshold=nms_thresh,
        )  # 进行非极大值抑制 (NMS)

        # 仅对发生变化的mask重新计算 RLE
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:  # 如果掩膜发生了变化
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]  # 更新 RLE
                mask_data["boxes"][i_mask] = boxes[i_mask]  # 更新边界框
        mask_data.filter(keep_by_nms)  # 根据 NMS 保留的索引过滤mask数据

        return mask_data

    def refine_with_m2m(self, points, point_labels, low_res_masks, points_per_batch):
        """
        使用 M2M (多掩膜生成器) 进行掩膜的进一步精化。

        参数：
          points (np.ndarray): 点坐标数组。
          point_labels (np.ndarray): 点的标签。
          low_res_masks (np.ndarray): 低分辨率掩膜。
          points_per_batch (int): 每批次的点数量。

        返回：
          Tuple[torch.Tensor, torch.Tensor]: 新生成的掩膜和对应的 IoU 预测。
        """
        new_masks = []      # 用于存储精化后的掩膜
        new_iou_preds = []  # 用于存储精化后的 IoU 预测

        for cur_points, cur_point_labels, low_res_mask in batch_iterator(
            points_per_batch, points, point_labels, low_res_masks
        ):
            # 对当前批次进行mask预测
            best_masks, best_iou_preds, _ = self.predictor._predict(
                cur_points[:, None, :],
                cur_point_labels[:, None],
                mask_input=low_res_mask[:, None, :],
                multimask_output=False,
                return_logits=True,
            )
            new_masks.append(best_masks)  # 收集掩码
            new_iou_preds.append(best_iou_preds)  # 收集 IoU 预测
        masks = torch.cat(new_masks, dim=0)  # 合并所有掩码
        return masks, torch.cat(new_iou_preds, dim=0)  # 合并所有 IoU 预测
