# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

# 检查用户是否从 sam2 仓库的父目录运行 Python
# （即克隆此仓库的目录）——这不被支持，因为这可能会导致 sam2 包被覆盖，并引发问题。
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # 如果用户的路径中包含 "sam2/sam2"，他们可能正在将仓库本身作为 "sam2" 导入
    # 而不是导入 "sam2" Python 包（即 "sam2/sam2" 目录）。
    # 这通常发生在用户从包含克隆的 sam2 仓库的父目录运行 Python 时。
    raise RuntimeError(
        "你可能正在从 sam2 仓库的父目录运行 Python "
        "（即克隆 https://github.com/facebookresearch/sam2 仓库的目录）。"
        "这不被支持，因为 `sam2` Python 包可能会被仓库名覆盖 "
        "(仓库也命名为 `sam2`，并包含 `sam2/sam2` 目录中的 Python 包)。"
        "请在另一个目录下运行 Python（例如从仓库目录而不是它的父目录，或从你的主目录）"
        "在安装 SAM 2 后再运行。"
    )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """
    根据给定的配置文件和检查点路径构建 SAM2 模型。

    参数：
    - config_file: 配置文件名。
    - ckpt_path: 可选的检查点路径。
    - device: 设备类型，例如 "cuda"。
    - mode: 模式，默认为 "eval"。
    - hydra_overrides_extra: 额外的 Hydra 覆盖选项。
    - apply_postprocessing: 是否应用后处理步骤。
    - **kwargs: 其他额外参数。

    返回：
    - model: 构建好的 SAM2 模型。
    """
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # 如果单一掩码不稳定，则动态回退到多掩码
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # 读取配置并初始化模型
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    # 设置 Hydra 覆盖选项，初始化 SAM2 视频预测器
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # 如果单一掩码不稳定，则动态回退到多掩码
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # 对与点击交互的帧在记忆编码器中的掩码使用 sigmoid，以便编码的掩码与用户点击的内容完全一致
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # 填补低分辨率掩膜中的小孔，直到 `fill_hole_area` （在调整为原始视频分辨率之前）
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # 读取配置并初始化模型
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)

def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        # 从检查点路径加载模型状态字典
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        # 加载状态字典到模型中，检查是否有缺失或意外的键
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
