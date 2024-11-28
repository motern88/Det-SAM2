# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # 仅在 Ampere (8.0) 或更新的 GPU 上使用 Flash Attention
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention 已禁用，因为它需要具有 Ampere (8.0) CUDA 功能的 GPU。",
                category=UserWarning,
                stacklevel=2,
            )
        # 对于 PyTorch 2.2 之前的版本保留数学内核（Flash Attention v2 仅在 PyTorch 2.2+ 中可用，
        # 而 Flash Attention v1 不能处理所有情况）
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"您正在使用 PyTorch {torch.__version__}，不支持 Flash Attention v2。"
                "考虑升级到 PyTorch 2.2+ 以获取 Flash Attention v2（可能更快）。",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    获取二值掩码的连接组件（8连通性），形状为 (N, 1, H, W)。

    输入：
    - mask: 形状为 (N, 1, H, W) 的二值掩码张量，其中 1 表示前景，0 表示背景。

    输出：
    - labels: 形状为 (N, 1, H, W) 的张量，包含前景像素的连接组件标签，背景像素为 0。
    - counts: 形状为 (N, 1, H, W) 的张量，包含前景像素的连接组件面积，背景像素为 0。
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    根据输入掩码计算边界框。

    输入：
    - masks: [B, 1, H, W] 的掩码，dtype=torch.Tensor

    返回：
    - box_coords: [B, 1, 4]，包含边界框的 (x, y) 坐标，即左上角和右下角的坐标，dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 是 JPEG 图像的预期数据类型
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"未知的图像数据类型: {img_np.dtype} 在 {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # 原始视频尺寸
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    一组视频帧异步加载的类，不阻塞会话启动。

    期望其异步加载同时能兼容不同的img_paths格式，包括：
    1.多帧图像文件路径的列表;2.单帧图像路径列表;3.多帧np数组列表;4.单帧np数组
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # `self.images` 中的项将被异步加载
        self.images = [None] * len(img_paths)
        # 捕获并在异步加载线程中引发任何异常
        self.exception = None
        # 视频高度和视频宽度将在加载第一张图像时填充
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        self._num_frames = len(img_paths)
        if self._num_frames > 0:
            # 加载第一帧以填充视频高度和视频宽度，并进行缓存（因为这是用户最可能点击的地方）
            first_image = self.__getitem__(0)
            channels, height, width = first_image.shape
            self._shape = (self._num_frames, channels, height, width)
        else:
            self._shape = (0, 0, 0, 0)  # 默认形状为0

        # 异步加载其余帧，不阻塞会话启动
        def _load_frames():
            try:
                # for n in tqdm(range(len(self.images)), desc="帧加载中"):
                for n in range(len(self.images)):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)  # TODO：注释掉尝试节约更多资源
        self.thread.start()  # TODO：注释掉尝试节约更多资源

    @property
    def shape(self):
        """返回 (num_frames, channels, height, width) 的元组。"""
        return self._shape

    def to_tensor(self):
        """将所有图像帧转换为 Tensor，并返回一个形状为 (num_frames, channels, height, width) 的 Tensor。"""
        frame_list = [self.__getitem__(i).clone().detach() for i in range(self._num_frames)]
        return torch.stack(frame_list)

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("帧加载线程中的失败") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img_path = self.img_paths[index]
        if isinstance(img_path, str):
            # img_path 是图像路径
            img, video_height, video_width = _load_img_as_tensor(img_path, self.image_size)
            self.video_height, self.video_width = video_height, video_width
        elif isinstance(img_path, np.ndarray):
            # img_path 是 np.ndarray 格式的帧
            img_np = cv2.resize(img_path, (self.image_size, self.image_size)) / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()
            self.video_height, self.video_width = img_path.shape[:2]
        else:
            raise TypeError(f"不支持的 img_paths 类型: {type(img_path)}")

        # img, video_height, video_width = _load_img_as_tensor(
        #     self.img_paths[index], self.image_size
        # )
        # self.video_height = video_height
        # self.video_width = video_width

        # 按均值和标准差进行归一化
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img  # TODO：注释掉尝试节约更多资源
        return img

    def __len__(self):
        return len(self.images)

# 将img_tensor复原成frame_rgb
def tensor_to_frame_rgb(
    tensor_img,
    original_size=(1920, 1080),
    img_mean=(0.485, 0.456, 0.406),  # 归一化参数需要保持和load_video_frames加载时一致！
    img_std=(0.229, 0.224, 0.225),  # 归一化参数需要保持和load_video_frames加载时一致！
):
    '''
    将标准化的 tensor 图像转换回 RGB 格式的 NumPy 数组。

    注意，我们期望tensor_to_frame_rgb尽量达到load_video_frames的逆操作，但实际resize和标准化以及逆标准化过程中不可避免带来精度损失。
    我们只能期望它不会显著影响视觉效果。
    '''
    # 将均值和标准差转为张量
    device = tensor_img.device
    img_mean = torch.tensor(img_mean, dtype=torch.float32, device=device)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32, device=device)[:, None, None]

    # 去除均值和标准差的归一化操作
    tensor_img = tensor_img * img_std + img_mean

    # 将图像的维度从 (1, 3, 1024, 1024) 转为 (3, 1024, 1024)，然后转为 (1024, 1024, 3)
    tensor_img = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 将图像从 1024x1024 调整回 1080x1920
    frame_rgb = cv2.resize(tensor_img, original_size)

    # 将图像从 0-1 范围转换回 0-255，并转换为 uint8 格式
    frame_rgb = np.clip(frame_rgb * 255, 0, 255).astype(np.uint8)

    return frame_rgb


# 一次性加载视频中所有帧
def load_video_frames(
    video_path,  # 可以传入包含图像帧的文件夹,也可以传入包含图像帧路径的列表
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,  # 是否异步加载帧
    compute_device=torch.device("cuda"),
):
    """
    从 JPEG 文件目录加载视频帧（格式为 "<frame_index>.jpg"）。

    帧被调整为 image_size x image_size 并加载到 GPU（如果 `offload_video_to_cpu` 为 `False`）
    或者加载到 CPU（如果 `offload_video_to_cpu` 为 `True`）。

    可以通过将 `async_loading_frames` 设置为 `True` 来异步加载帧。
    """

    if isinstance(video_path, str) and os.path.isdir(video_path):
        # print("如果传入的是图像帧的文件夹目录")
        # 如果传入的是图像帧的文件夹目录
        jpg_folder = video_path
        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"在 {jpg_folder} 中未找到图像")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]

    elif isinstance(video_path, list) and all(os.path.isfile(p) for p in video_path):
        # print("如果传入的是图像文件路径的列表")
        # 如果传入的是图像文件路径的列表
        img_paths = video_path
        num_frames = len(img_paths)

    elif isinstance(video_path, np.ndarray):
        # # 视频流中已加载的帧，假设每一帧是RGB格式的np.ndarray
        frame_rgb = video_path
        num_frames = 1  # 处理一帧

    elif isinstance(video_path, list) and all(isinstance(p, np.ndarray) for p in video_path):
        # print(f"视频流中累积的以列表形式存储的多帧，假设每一帧是RGB格式的np.ndarray")
        # 视频流中累积的以列表形式存储的多帧，假设每一帧是RGB格式的np.ndarray
        frame_rgb_list = video_path
        num_frames = len(frame_rgb_list)

    elif isinstance(video_path, str) and os.path.isfile(video_path):
        # print("如果传入的是单个图像文件路径")
        # 如果传入的是单个图像文件路径
        img_paths = [video_path]
        num_frames = 1

    else:
        print("传入不支持的帧格式")
        raise NotImplementedError(
            "目前仅支持 JPEG 帧。对于视频文件，您可以使用 ffmpeg (https://ffmpeg.org/) "
            "将帧提取到 JPEG 文件夹中，例如：\n"
            "```\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
            "```\n"
            "其中 `-q:v` 生成高质量 JPEG 帧，`-start_number 0` 要求 "
            "ffmpeg 从 00000.jpg 开始 JPEG 文件。"
            "随后video_path支持传入文件夹路径，或包含图像路径的列表，或单个图像路径"
        )

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # 此时img_paths可能格式有：1.多帧图像文件路径的列表;2.单帧图像路径列表;3.多帧np数组列表;4.单帧np数组
    if async_loading_frames:
        if "frame_rgb" in locals():
            async_frames = frame_rgb
        elif "frame_rgb_list" in locals():
            async_frames = frame_rgb_list
        else:
            async_frames = img_paths

        lazy_images = AsyncVideoFrameLoader(
            async_frames,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    # 原官方代码此处是转为FP32精度,可以尝试将其以FP16精度加载以节省一半内存开销
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)  # dtype=torch.float32

    if "frame_rgb" in locals():
        # 处理视频流中的帧，手动处理并归一化
        img_np = cv2.resize(frame_rgb, (image_size, image_size)) / 255.0
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        images[0] = img
        video_height, video_width = frame_rgb.shape[:2]
    elif "frame_rgb_list" in locals():
        # 处理视频流中累积的多个帧，手动处理并归一化
        for n, frame_rgb in enumerate(frame_rgb_list):
            # print(frame_rgb.shape) # (1080, 1920, 3)
            img_np = cv2.resize(frame_rgb, (image_size, image_size)) / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)
            images[n] = img
        video_height, video_width = frame_rgb_list[0].shape[:2]
    else:
        if len(img_paths) == 0:
            print("sam2.utils.misc.load_video_frames()中传入了错误的图像帧")
            return None, None, None
        # 处理图像路径列表或文件夹中的帧，使用SAM2自带的_load_img_as_tensor
        for n, img_path in enumerate(tqdm(img_paths, desc="帧加载中")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)

    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    # 按均值和标准差进行归一化
    images -= img_mean
    images /= img_std

    # print("images", images.shape)  # torch.Size([1, 3, 1024, 1024])
    # print(video_height, video_width)  # 1080 1920
    return images, video_height, video_width

def fill_holes_in_mask_scores(mask, max_area):
    """
    后处理步骤，用于填充掩码分数中小于 `max_area` 的小孔。
    """
    # 孔是背景中连接组件的区域，面积 <= self.max_area
    # （背景区域是掩码分数 <= 0 的区域）
    assert max_area > 0, "max_area 必须为正数"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # 使用小的正掩码分数 (0.1) 填充孔，将它们变为前景。
        mask = torch.where(is_hole, 0.1, mask)

    # except Exception as e:
    #     # 如果 CUDA 内核失败，则跳过填充小孔的后处理步骤
    #     warnings.warn(
    #         f"{e}\n\n由于上述错误，跳过后处理步骤。您仍然可以使用 SAM 2，"
    #         "忽略上述错误是可以的，尽管一些后处理功能可能会受到限制（这不会影响大多数情况下的结果；请参见 "
    #         "https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md）。",
    #         category=UserWarning,
    #         stacklevel=2,
    #     )
    except Exception:

        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """将新点和标签添加到之前的点输入（添加到末尾）。"""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


if __name__ == '__main__':
    '''
    验证tensor_to_frame_rgb与_load_img_as_tensor是否接近互为逆操作
    '''

    # 定义原始图像加载的大小和标准化参数
    image_size = 1024
    original_size = (1920, 1080)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    # 生成随机图像并转换为 uint8 格式
    original_frame_rgb = np.random.randint(0, 256, (original_size[0], original_size[1], 3), dtype=np.uint8)

    # 使用 load_video_frames 的逻辑模拟标准化和调整大小
    original_frame_resized = cv2.resize(original_frame_rgb, (image_size, image_size)) / 255.0
    img_tensor = torch.from_numpy(original_frame_resized).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # 保存图像到磁盘，方便 `load_video_frames` 使用
    test_image_path = 'test_random_image.jpg'
    cv2.imwrite(test_image_path, cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR))

    # 调用 load_video_frames 进行加载和标准化
    images, video_height, video_width = load_video_frames(
        video_path=[test_image_path],
        image_size=image_size,
        offload_video_to_cpu=True,
        img_mean=img_mean,
        img_std=img_std
    )

    # 从 tensor 恢复到 RGB 格式的图像
    restored_frame_rgb = tensor_to_frame_rgb(
        images[0].unsqueeze(0),
        original_size=original_size,
        img_mean=img_mean,
        img_std=img_std
    )

    # 计算原始图像与恢复图像的差异
    original_frame_resized = cv2.resize(original_frame_rgb, original_size)
    difference = np.abs(original_frame_resized.astype(np.float32) - restored_frame_rgb.astype(np.float32))
    mean_difference = np.mean(difference)
    max_difference = np.max(difference)

    print(f"平均像素差异: {mean_difference:.2f}")  # 32.09
    print(f"最大像素差异: {max_difference:.2f}")  # 180.00

    # 删除测试图像文件
    os.remove(test_image_path)