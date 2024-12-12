# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch
import gc

from sympy.physics.units import current
from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames, AsyncVideoFrameLoader

import subprocess  # TODO;暂时开发用于查看内存占用，后续删除

class SAM2VideoPredictor(SAM2Base):
    """
    Predictor class for handling user interactions and managing inference state.
    处理用户交互和管理推理状态的预测器类。
    """

    def __init__(
        self,
        fill_hole_area=0,
        # Whether to apply non-overlapping constraints on output object masks
        # 是否对输出对象mask应用无重叠约束
        non_overlap_masks=False,
        # Whether to clear non-conditional memory of surrounding frames after adding correction clicks
        # 添加修正点击后是否清除周围帧的非条件记忆（可能包含过时信息）；
        # Note: This only applies to *single object tracking*, unless `clear_non_cond_mem_for_multi_obj` is also set to True.
        # 注意：这仅适用于*单对象跟踪*，除非 `clear_non_cond_mem_for_multi_obj` 也设置为 True）
        clear_non_cond_mem_around_input=False,
        # Whether to also clear non-conditional memory of surrounding frames (only effective if `clear_non_cond_mem_around_input` is True).
        # 是否还清除周围帧的非条件记忆（仅在 `clear_non_cond_mem_around_input` 为 True 时有效）。
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj

    # Initialize inference state / 初始化inference_state
    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=True,  # Whether to offload video frames to CPU memory to reduce GPU memory usage by 0.025GB per frame.
        # True 将视频帧转移到 CPU 内存中,可以降低每帧0.025G显存开销,减小2.5G开销/100帧
        offload_state_to_cpu=False,  # For speed, set to False; for memory savings, set to True.
        # 为了速度可以设置为False,节约显存可以设置为True ！True时所有涉及storage_device张量设备迁移的需要设置non_blocking=False否则掩码输出错位 ;将['output_dict]和['output_dict_per_obj']中张量存在cpu中, 时间开销增加约22%
        async_loading_frames=False,  # Async loading compatibility; async loading seems to have no impact on memory usage.
        # False 实现AsyncVideoFrameLoader兼容nd数组的格式输入,但是异步加载似乎没有影响最终显存和内存开销
    ):
        print(f"Initializing: offload_video_to_cpu:{offload_video_to_cpu},offload_state_to_cpu:{offload_state_to_cpu}")

        compute_device = self.device  # Device where the model resides / 模型所在的设备
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images  # Store video frames / 存储视频帧图像 (N, 3, 1024, 1024)

        # Number of video frames N, which may serve as a marker for the maximum frame index,
        # ensuring it always represents the total number of historically loaded video frames rather than the current video frame count
        # 视频帧的数量 N ,可能起到标记最大帧索引的作用, 需要保证其一直为所有历史加载过的视频帧数而非当前视频帧数
        inference_state["num_frames"] = len(images)  # (特定情况会删减旧的帧释放内存，故会出现历史加载过的视频帧数不等于当前视频帧数的情况)

        # Records the index of tensors in images corresponding to the actual video frame index [0,1,4,5,6,9,10,11,...] where index N's number K represents
        # the Kth frame of the actual video corresponding to the Nth tensor in images
        # 记录images中tensor的索引对应真实的视频帧索引 [0,1,4,5,6,9,10,11,...]其中N索引上的数字K代表images中第N个tensor对应真实视频的第K帧
        inference_state["images_idx"] = list(range(len(images)))

        # Whether to transfer video frames to CPU memory
        # 是否将视频帧转移到 CPU 内存中，启用此选项可以节省 GPU 内存，只有非常小的开销
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu

        # Whether to transfer the inference state to CPU memory, enabling this option can save GPU memory, but will reduce the FPS of tracking
        # 是否将推理状态转移到 CPU 内存中，启用此选项可以节省 GPU 内存，但会降低跟踪的 FPS
        # （例如，在 768x768 模型的测试中，跟踪一个对象时 FPS 从 27 降至 24，跟踪两个对象时从 24 降至 21）
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu

        # The original height and width of the video, used to adjust the final output scores
        # 原始视频的高度和宽度，用于调整最终输出的分数
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device  # compute device / 计算设备
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")  # Storage device set to CPU /存储设备设置为 CPU
        else:
            inference_state["storage_device"] = compute_device  # Storage device set to compute device / 存储设备设置为计算设备
        # Per-frame input / 每帧的输入
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # A small number of recently accessed frames' visual features for quick interaction
        # 少量最近访问帧上的视觉特征，用于快速交互
        inference_state["cached_features"] = {}
        # Values that remain constant across all frames (so we only need to save one copy)
        # 在所有帧中保持不变的值（所以我们只需要保存一份）
        inference_state["constants"] = {}
        # Mapping between client object IDs and model object indices
        # 客户端对象 ID 和模型对象索引之间的映射
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []

        # Store the model's tracking results and states on each frame. Savings:
        # The dictionaries ["maskmem_features"] and ["pred_masks"] in the dictionary ["cond_frame_outputs"] are both stored on the storage_device
        # 存储模型在每帧上的跟踪结果和状态。节省开销：
        # 字典["cond_frame_outputs"]中["maskmem_features"]和["pred_masks"]均存储在storage_device上
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # Dictionary containing {frame_idx: <out>}
        }
        # # Slices (views) of each object's tracking results, sharing the same memory as "output_dict". It is ensured that ["maskmem_features"] is stored on the storage_device
        # 每个对象跟踪结果的切片（视图），与 "output_dict" 共享相同的内存。已确保其中["maskmem_features"]存储在storage_device上
        inference_state["output_dict_per_obj"] = {}

        # Temporary storage, where new outputs are stored when users interact with frames (e.g., adding clicks or masks)
        # Before propagation begins, they are merged into "output_dict"
        # 临时存储，当用户与帧交互时（例如添加点击或mask），新输出会存储在此
        # 在传播开始之前会合并到 "output_dict" 中
        inference_state["temp_output_dict_per_obj"] = {}

        # Frames that already include the merged outputs from clicks or mask inputs
        # (We use their merged outputs directly during tracking)
        # 已经包含从点击或掩膜输入合并后的输出的帧
        # （我们在跟踪过程中直接使用它们的合并输出）
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # 包含帧索引的集合
            "non_cond_frame_outputs": set(),  # 包含帧索引的集合
        }

        # Metadata for each tracking frame (e.g., tracking direction)
        # 每个跟踪帧的元数据（例如，跟踪方向）
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # # Preload all conditional frame indices and all non-conditional frame indices in the preload memory bank
        # 预加载内存库中所有条件帧索引和所有非条件帧索引
        inference_state["preloading_memory_cond_frame_idx"] = None
        inference_state["preloading_memory_non_cond_frames_idx"] = None
        # The maximum update length for updating historical frame information when new client IDs are added during tracking
        # 如果在追踪过程中出现新增客户端ID，进行更新历史帧信息时的最大更新长度
        inference_state["max_update_length_for_new_obj_id"] = 100
        # Preheat the visual backbone network and cache the image features of the 0th frame
        # 预热视觉骨干网络，并缓存第 0 帧的图像特征
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    # Reset the inference_state settings in the preload memory library (e.g., memory optimization settings)
    # 重置预加载内存库中的inference_state设置(例如内存优化相关设置)
    def init_preloading_state(
        self,
        inference_state,
        offload_video_to_cpu = True,  # True will transfer video frames to CPU memory, reducing memory consumption by 0.025G per frame, reducing 2.5G consumption/100 frames
        # True 将视频帧转移到 CPU 内存中,可以降低每帧0.025G显存开销,减小2.5G开销/100帧
        offload_state_to_cpu = True, # !True requires setting non_blocking=False for all tensor device migrations involving storage_device, otherwise mask output will be misaligned; storing tensors in cpu increases time overhead by about 22%
        # ！True时所有涉及storage_device张量设备迁移的需要设置non_blocking=False否则掩码输出错位 ;将['output_dict]和['output_dict_per_obj']中张量存在cpu中, 时间开销增加约22%
    ):
        '''
        This method is called only when an external preload memory library is loaded,
        and is used to synchronize the preload memory library's settings (such as storage optimization parameters)
        to the latest settings for this inference.
        该方法仅在外部加载预加载内存库时调用，用于将预加载内存库的相关设置（例如存储优化参数）同步为本次推理的最新设置
        '''
        if offload_video_to_cpu:
            # Transfer video frames to CPU memory
            inference_state["images"] = inference_state["images"].to("cpu")  # 将视频帧转移到 CPU 内存中

        compute_device = self.device  # The device where the model is located / 模型所在的设备
        if offload_state_to_cpu:
            # Set storage device to CPU
            inference_state["storage_device"] = torch.device("cpu")  # 存储设备设置为 CPU
        else:
            # Set storage device to compute device
            inference_state["storage_device"] = compute_device  # 存储设备设置为计算设备
        device = inference_state["storage_device"]

        for frame_idx in range(inference_state["num_frames"]-1):
            # print(f"处理预加载内存库中{frame_idx}")
            # Process each conditional frame in the memory bank (memory bank only contains conditional frames)
            # 内存库中只应存在条件帧
            current_cond_frame = inference_state['output_dict']['cond_frame_outputs'][frame_idx]

            current_cond_frame['maskmem_features'] = current_cond_frame['maskmem_features'].to(device,non_blocking=False)
            current_cond_frame['pred_masks'] = current_cond_frame['pred_masks'].to(device,non_blocking=False)

            for obj_idx in inference_state["obj_idx_to_id"].keys():
                current_cond_obj = inference_state['output_dict_per_obj'][obj_idx]['cond_frame_outputs'][frame_idx]

                current_cond_obj['maskmem_features'] = current_cond_obj['maskmem_features'].to(device,non_blocking=False)
                current_cond_obj['pred_masks'] = current_cond_obj['pred_masks'].to(device,non_blocking=False)

        print("Preload memory bank processed / 预加载内存库处理完毕")

    # Add new frames to the existing inference_state
    # 在已有inference_state添加新帧
    @torch.inference_mode()
    def update_state(
        self,
        video_path,
        inference_state,  # Add new frames and update inference_state / 添加新帧，更新到inference_state
        async_loading_frames=False,
    ):
        '''
        Use init_state method for the first time to add frames, and use this method for subsequent frame additions.
        首次添加帧使用init_state方法，之后再次添加帧仅使用此方法
        '''
        # Load new video frames / 加载新的视频帧
        new_images, new_video_height, new_video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=inference_state["offload_video_to_cpu"],
            async_loading_frames=async_loading_frames,
            compute_device=self.device,
        )

        # Get the original video height and width / 获取原始视频的高度和宽度
        video_height = inference_state["video_height"]
        video_width = inference_state["video_width"]
        assert video_height == new_video_height and video_width == new_video_width, "Ensure new image frames and existing image frames have the same height and width / 确保新图像帧和已有图像帧具有相同高度和宽度)"

        # Merge new and existing frame index mapping lists / 合并新的和现有的帧索引映射列表
        last_frame_index = inference_state["images_idx"][-1]
        new_frame_indices = list(range(last_frame_index+1, last_frame_index+1 + len(new_images)))
        inference_state["images_idx"].extend(new_frame_indices)

        # Merge new frames with existing frames / 合并新帧和现有帧
        images = inference_state["images"]  # Get previous video frame images / 获取此前视频帧图像
        assert images.shape[1:] == new_images.shape[1:], "Ensure new image frames and existing image frames have the same dimensions (channels, height, width) / 确保新图像帧和已有图像帧具有相同的维度 (通道数, 高度, 宽度)"
        # Check if images and new_images are instances of AsyncVideoFrameLoader for asynchronous loading
        # 检查 images 和 new_images 是否为 AsyncVideoFrameLoader 异步加载的实例
        if isinstance(images, AsyncVideoFrameLoader):
            images = images.to_tensor()
        if isinstance(new_images, AsyncVideoFrameLoader):
            new_images = new_images.to_tensor()
        combined_images = torch.cat((images, new_images), dim=0)  # torch.Size([N frames, 3, 1024, 1024])

        #  print("---update_state:combined_images.shape:",combined_images.shape)

        # Update the inference state / 更新推理状态
        inference_state["images"] = combined_images  #  Store video frame images / 存储视频帧图像

        # The number of video frames, ensure it always represents the total number of historically loaded video frames rather than the current video frame count, otherwise errors may occur
        # 视频帧的数量,需要保证其一直为所有历史加载过的视频帧数而非当前视频帧数,否则可能会引发报错
        inference_state["num_frames"] += len(new_images)
        # print("total num_frames",inference_state["num_frames"])

        return inference_state


    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        from sam2.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)  # 加载并构建模型
        return sam_model  # 返回加载的模型实例

    def _obj_id_to_idx(self, inference_state, obj_id):
        """
        Map client object ID to model object index.
        将客户端对象 ID 映射到模型端对象索引。
        """
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)  # get obj_id / 获取对象索引
        if obj_idx is not None:
            return obj_idx  # 如果对象 ID 已存在，直接返回对应的对象索引

        #  Determine if new objects can be added during non-tracking phases
        allow_new_object = not inference_state["tracking_has_started"]  # 判断是否是非跟踪过程中添加新对象
        # Adding new objects before tracking starts.
        if allow_new_object:  # 跟踪未开始时添加新的类别
            # 获取下一个对象插槽
            obj_idx = len(inference_state["obj_id_to_idx"])  # Assign a new object index / 分配新的对象索引
            inference_state["obj_id_to_idx"][obj_id] = obj_idx  # Update mapping from object ID to index / 更新对象 ID 到索引的映射
            inference_state["obj_idx_to_id"][obj_idx] = obj_id  # Update mapping from index to object ID / 更新对象索引到 ID 的映射
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])  # Update the list of object IDs / 更新对象 ID 列表
            # 为此对象设置输入和输出结构
            inference_state["point_inputs_per_obj"][obj_idx] = {}  # Initialize input and output structures for the object / 初始化此对象的点输入结构
            inference_state["mask_inputs_per_obj"][obj_idx] = {}  # Initialize mask input structure for the object / 初始化此对象的掩膜输入结构
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
                "non_cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
                "non_cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
            }
            return obj_idx  # Return the newly assigned object index / 返回新分配的对象索引

        # adding new objects after tracking starts
        # 在跟踪开始后添加新对象
        else:
            # 获取下一个对象插槽
            obj_idx = len(inference_state["obj_id_to_idx"])  # Assign a new object index / 分配新的对象索引
            inference_state["obj_id_to_idx"][obj_id] = obj_idx  # Update mapping from object ID to index / 更新对象 ID 到索引的映射
            inference_state["obj_idx_to_id"][obj_idx] = obj_id  # Update mapping from index to object ID / 更新对象索引到 ID 的映射
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])  # Update the list of object IDs / 更新对象 ID 列表
            # Initialize input and output structures for the object
            # 为此对象设置输入和输出结构
            inference_state["point_inputs_per_obj"][obj_idx] = {}  # Initialize point input structure for the object / 初始化此对象的点输入结构
            inference_state["mask_inputs_per_obj"][obj_idx] = {}  # Initialize mask input structure for the object / 初始化此对象的掩膜输入结构
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
                "non_cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
                "non_cond_frame_outputs": {},  # 包含 {frame_idx: <out>} 的字典
            }

            def print_consolidated_out(consolidated_out):
                for key, value in consolidated_out.items():
                    if isinstance(value, torch.Tensor):
                        print(f"Key: {key}, Value: Tensor, Shape: {value.shape}")
                    elif isinstance(value, dict):
                        print(f"Key: {key}, Value: dict")
                        print_consolidated_out(value)  # 递归打印嵌套的字典
                    elif isinstance(value, list):
                        print(f"Key: {key}, Value: list, Length: {len(value)}")
                    else:
                        print(f"Key: {key}, Value: {value}")

            # List of condition frame indices in preloaded memory.
            preloading_memory_cond_frame_idx = inference_state["preloading_memory_cond_frame_idx"]  # 预加载内存库中条件帧索引列表
            # Maximum length of frames to update.
            max_update_length = inference_state["max_update_length_for_new_obj_id"]  # 获取需要更新的最大长度
            print(f"New client ID encountered during tracking. Updating information for the most recent {max_update_length} frames "
                  f"and preloaded memory (if applicable) with the latest ID mapping standards."
                  f" / 跟踪过程中出现新的客户端ID，正在以最新ID映射标准更新内存库中最近{max_update_length}帧信息和预加载内存库信息(如果存在)")

            output_dict = inference_state["output_dict"]
            # Retrieve all conditional frame indices in chronological order.
            cond_frame_indices = sorted(output_dict["cond_frame_outputs"].keys())  # 获取所有条件帧索引并按时间顺序排序
            # Only keep the most recent `max_update_length` frame indices.
            # 只选择最近的 max_update_length 帧索引
            if max_update_length > 0:
                cond_frame_indices = cond_frame_indices[-max_update_length:]
            # Add conditional frames from preloaded memory to the update queue.
            # 添加预加载内存库中的条件帧进入更新队列
            if preloading_memory_cond_frame_idx is not None:
                for t in preloading_memory_cond_frame_idx:
                    if t not in cond_frame_indices:
                        cond_frame_indices.append(t)
            # Update all historical condition frames using the latest mapping standards.
            # 以新的映射标准更新所有历史条件帧
            for cond_frame_idx in tqdm(cond_frame_indices,desc=f"更新最近{max_update_length}帧和预加载内存库内的条件帧"):
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx=cond_frame_idx,
                    is_cond=True,
                    run_mem_encoder=True,
                    consolidate_at_video_res=False,
                )
                # print_consolidated_out(consolidated_out)
                # Merge them into "output_dict".
                # 将它们合并到 "output_dict" 中

                output_dict["cond_frame_outputs"][cond_frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, cond_frame_idx, consolidated_out, storage_key="cond_frame_outputs"
                )
            # # 以新的映射标准更新所有历史非条件帧 TODO：更新非条件帧是否必要？注释掉似乎也能跑，如无必要则可以省这一步，减少计算开销
            # non_cond_frame_outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
            # for non_cond_frame_idx in tqdm(non_cond_frame_outputs.keys(),desc="更新所有历史非条件帧"):
            #     consolidated_out = self._consolidate_temp_output_across_obj(
            #         inference_state,
            #         frame_idx=non_cond_frame_idx,
            #         is_cond=False,
            #         run_mem_encoder=True,
            #         consolidate_at_video_res=False,
            #     )
            #     # print_consolidated_out(consolidated_out)
            #     # 将它们合并到 "output_dict" 中
            #     output_dict["non_cond_frame_outputs"][non_cond_frame_idx] = consolidated_out
            #     self._add_output_per_object(
            #         inference_state, non_cond_frame_idx, consolidated_out, storage_key="non_cond_frame_outputs"
            #     )
            return obj_idx  # 返回新分配的对象索引

            # raise RuntimeError(
            #     f"在跟踪开始后无法添加新对象 ID {obj_id}。 "
            #     f"所有现有的对象 ID: {inference_state['obj_ids']}。"
            #     f"请调用 'reset_state' 重新开始。"
            # )  # 如果跟踪已经开始，抛出错误

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """
        Maps object index from the model to the object ID on the client side.
        将模型端的对象索引映射到客户端的对象 ID。
        """
        return inference_state["obj_idx_to_id"][obj_idx]  # 返回与给定对象索引对应的客户端对象 ID

    def _get_obj_num(self, inference_state):
        """
        Gets the total number of unique object IDs received so far in this session
        获取此会话中迄今为止接收到的唯一对象 ID 的总数
        """
        return len(inference_state["obj_idx_to_id"])  # 返回对象 ID 的总数量

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,  # Index of the current frame / 当前帧的索引
        obj_id,  # Object ID provided by the client / 客户端提供的对象 ID
        points=None,  # 添加的点（坐标），默认为 None
        labels=None,  # 每个点的标签（例如[1,0,0,0,1] 等,1正提示,0负提示），默认为 None
        # Whether to clear old points, default is True
        clear_old_points=True,  # 是否清除旧点，默认为 True
        # Whether to normalize point coordinates, default is True
        normalize_coords=True,  # 是否对点坐标进行归一化，默认为 True
        # Box (coordinates) to be added, default is None
        box=None,  # 要添加的框（坐标），默认为 None
    ):
        """
        Adds new point prompts to the frame.
        为帧添加新的点提示。
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)  # Map client object ID to model-side object index / 将客户端对象 ID 映射到模型端对象索引
        # print(f"add_new_points_or_box帧索引：{frame_idx},客户端obj_id：{obj_id}的框提示,模型端obj_idx:{obj_idx}")
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]  # Get point inputs for the corresponding frame / 获取对应帧的点输入
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]  # Get mask inputs for the corresponding frame / 获取对应帧的掩码输入
        # print(f"开始时point_inputs_per_frame:{point_inputs_per_frame}")
        # print(f"开始时mask_inputs_per_frame:{mask_inputs_per_frame}")

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("At least points or box must be provided as input")

        if points is None:  # 如果没有提供点，创建一个空的点张量
            points = torch.zeros(0, 2, dtype=torch.float32)  # Create an empty tensor for points if not provided
        elif not isinstance(points, torch.Tensor):  # 如果点不是张量，将其转换为张量
            points = torch.tensor(points, dtype=torch.float32)   # Convert points to a tensor if it's not already
        if labels is None:  # 如果没有提供标签，创建一个空的标签张量
            labels = torch.zeros(0, dtype=torch.int32)   # Create an empty tensor for labels if not provided
        elif not isinstance(labels, torch.Tensor):  # 如果标签不是张量，将其转换为张量
            labels = torch.tensor(labels, dtype=torch.int32)    # Convert labels to a tensor if it's not already
        if points.dim() == 2:  # 如果点张量是二维的，添加批次维度
            points = points.unsqueeze(0)  # Add a batch dimension if the points tensor is 2D
        if labels.dim() == 1:  # 如果标签张量是一维的，添加批次维度
            labels = labels.unsqueeze(0)  # Add a batch dimension if the labels tensor is 1D

        # If a box is provided, add it as the first two points and set labels to 2 and 3
        # This is consistent with the training method of SAM 2
        # 如果提供了 box，我们将其添加为前两个点，并设置标签为 2 和 3 ，这与 SAM 2 的训练方式一致
        if box is not None:
            # If old points are not cleared, raise an exception because box must be added before points
            if not clear_old_points:  # 如果没有清除旧点，抛出异常，因为 box 必须在点之前添加
                raise ValueError(
                    "You cannot add a box without clearing old points, because the box prompt must be provided before point prompts (please use clear_old_points=True) / "
                    "不能在不清除旧点的情况下添加 box，因为 box 提示必须在点提示之前提供 "
                    "(请使用 clear_old_points=True)"
                )
            # if inference_state["tracking_has_started"]:  # 如果跟踪已经开始，发出警告，提示 box 可能无法有效融合
            #     warnings.warn(
            #         "在跟踪开始后添加 box。SAM 2 可能无法始终有效地将 box 提示用于精炼。如果您打算 "
            #         "在跟踪开始前使用 box 提示，请调用 'reset_state' 来重置推理状态。",
            #         category=UserWarning,
            #         stacklevel=2,
            #     )
            # If box is not a tensor, convert it to a tensor
            if not isinstance(box, torch.Tensor):  # 如果 box 不是张量，将其转换为张量
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)  # Reshape the box to (1, 2, 2) / 将 box 重塑为 (1, 2, 2) 的形状
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)  # Create box labels / 创建 box 标签
            box_labels = box_labels.reshape(1, 2)  # Reshape the labels / 调整标签的形状
            points = torch.cat([box_coords, points], dim=1)  # Concatenate box points with other points / 将 box 点与其他点连接
            labels = torch.cat([box_labels, labels], dim=1)  # Concatenate box labels with other labels / 将 box 标签与其他标签连接

        if normalize_coords:  # If normalization of coordinates is required / 如果需要对坐标进行归一化
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)  # Normalize the points / 对点进行归一化
        # 根据模型的内部图像大小对坐标进行缩放
        points = points * self.image_size
        points = points.to(inference_state["device"])  # Move the points to the inference device / 将点移动到推理设备
        labels = labels.to(inference_state["device"])  # Move the labels to the inference device / 将标签移动到推理设备

        if not clear_old_points:  # If old points are not cleared, get the current frame's point inputs / 如果不清除旧点，获取当前帧的点输入
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None  # Otherwise, clear the point inputs / 否则清除点输入
        point_inputs = concat_points(point_inputs, points, labels)  # Concatenate the new and old point inputs / 连接新旧点输入

        point_inputs_per_frame[frame_idx] = point_inputs  # Update the point inputs for the frame / 更新帧的点输入
        mask_inputs_per_frame.pop(frame_idx, None)  # Remove the old mask inputs / 移除旧的掩码输入

        # print(f"过程中point_inputs_per_frame:{point_inputs_per_frame}")
        # print(f"过程中mask_inputs_per_frame:{mask_inputs_per_frame}")

        # If this frame has not been tracked before, we treat it as an initial condition frame,
        # This means the input points will be used to generate segmentation results on this frame, without using the memory of other frames (like in SAM).
        # Otherwise, if it has been tracked, the input points will be used to correct the already tracked mask.
        # 如果此帧之前没有被跟踪过，我们将其视为初始条件帧，
        # 这意味着输入点将用于在该帧上生成分割结果，而不使用其他帧的记忆（如同在 SAM 中）。
        # 否则（如果已被跟踪），输入点将用于校正已经跟踪的掩码。
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # print(f"is_init_cond_frame:{is_init_cond_frame}")  # True

        # Whether to track in reverse time order / 是否按相反的时间顺序跟踪
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]  # Get the output dictionary for the object / 获取对象的输出字典
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]  # Get the temporary output dictionary for the object / 获取对象的临时输出字典

        # # 打印 cond_frame_outputs
        # def print_selected_structure(d, key_to_print="maskmem_features"):
        #     for main_key, frame_dict in d.items():
        #         if isinstance(frame_dict, dict):
        #             if key_to_print in frame_dict:
        #                 value = frame_dict[key_to_print]
        #                 if isinstance(value, torch.Tensor):
        #                     print(f"{main_key}: {{{key_to_print}: {value.shape}}}")
        #                 else:
        #                     print(f"{main_key}: {{{key_to_print}: {type(value)}}}")
        #         else:
        #             print(f"Unexpected structure at key: {main_key}")
        # print("obj_output_dict字典中cond_frame_outputs：")
        # print_selected_structure(obj_output_dict["cond_frame_outputs"])
        # 0: {maskmem_features: torch.Size([1, 64, 64, 64])}
        # 15: {maskmem_features: torch.Size([1, 64, 64, 64])}
        # 45: {maskmem_features: torch.Size([1, 64, 64, 64])}

        # print("obj_temp_output_dict字典中cond_frame_outputs：")
        # print_selected_structure(obj_temp_output_dict["cond_frame_outputs"])  # 空的
        # # 打印 non_cond_frame_outputs
        # print("non_cond_frame_outputs：")
        # print_selected_structure(obj_output_dict["non_cond_frame_outputs"])

        # If it's an initial condition frame or the model treats all frames with clicks/masks as condition frames, add the frame to the condition frame list
        # 如果是初始条件帧或模型视所有帧接收到点击/掩码作为条件帧，添加帧到条件输出
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get the previously predicted mask logits for this object and input them into the SAM mask decoder along with the new clicks
        # 获取此对象上先前预测的掩码 logits，并将其与新点击一起输入 SAM 掩码解码器
        prev_sam_mask_logits = None
        # First, look for the temporary output dictionary containing the latest output
        # 首先查找临时输出字典，其中包含最新输出
        # （如果未找到，则查找条件和非条件帧输出）
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        # If previous output exists and predicted masks are available
        # 如果存在先前的输出，并且存在预测的掩码
        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            # Move the previous mask logits to the correct device, avoiding blocking
            # 将之前的掩码 logits 转移到正确的设备上，并避免阻塞
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Limit the scale of prev_sam_mask_logits to avoid rare numerical issues
            # 限制 prev_sam_mask_logits 的尺度，避免罕见的数值问题
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        # Run single-frame inference and return the output for the current frame
        # 运行单帧推理，返回当前帧的输出
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # 处理单个对象的输出切片
            frame_idx=frame_idx,
            batch_size=1,  # 处理单个对象的输出切片
            is_init_cond_frame=is_init_cond_frame,  # 是否是初始条件帧
            point_inputs=point_inputs,  # 当前帧的输入点
            mask_inputs=None,  # 当前帧的掩码输入为空
            reverse=reverse,  # 是否按反向时间顺序进行跟踪
            # Skip memory encoder when adding clicks or masks. We run memory encoder at the beginning of `propagate_in_video` (after user completes clicks).
            # This allows us to enforce non-overlapping constraints on all objects before encoding them into memory.
            # 添加点击或掩码时跳过内存编码器。我们在 `propagate_in_video` 的开头执行memory encoder内存编码器（在用户完成点击后）。
            # 这使我们能够在将对象编码到内存之前，强制对所有对象执行非重叠约束。
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,  # 之前的掩码 logits 作为当前推理的输入
        )
        # Add the output to the output dictionary (for future memory use)
        # 将输出添加到输出字典中（以便将来用作记忆）
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # print(f"current_out:{current_out}")
        # 'maskmem_features': None, 'maskmem_pos_enc': None, 'pred_masks': 有值, 'obj_ptr': 有值

        # Adjust the output masks to the original video resolution
        # 将输出掩码调整到原始视频分辨率
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,  # 当前帧索引
            is_cond=is_cond,  # 是否是条件帧
            run_mem_encoder=False,  # 不运行内存编码器
            consolidate_at_video_res=True,  # 在视频分辨率下整合输出
        )
        # Get the output masks at the original video resolution
        # 获取原始视频分辨率下的输出掩码
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        # Return the current frame index, object ID list, and masks at video resolution
        # 返回当前帧索引，对象 ID 列表和视频分辨率下的掩码
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """
        Deprecated method. Please use `add_new_points_or_box` instead.
        弃用的方法。请使用 `add_new_points_or_box` 代替。
        """
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """
        Add a new mask for a specific frame.
        为某一帧添加新的掩码。
        """
        # Get the object index corresponding to the object ID
        # 根据对象ID获取对应的对象索引
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        # Get the point inputs and mask inputs for the current frame
        # 获取当前帧的点输入和掩码输入
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        # If the mask is not a tensor, convert it to a boolean tensor
        # 如果掩码不是张量类型，将其转换为布尔类型的张量
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        # Ensure the mask is 2D
        # 确保掩码是二维的
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        # Add batch and channel dimensions to the mask
        mask_inputs_orig = mask[None, None]  # 为掩码添加批次和通道维度
        # Convert mask to float and move to device
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])  # 将掩码转换为浮点型并移动到指定设备上

        # If the mask size does not match the model's image size, resize it
        # 如果掩码的尺寸与模型的图像尺寸不匹配，则进行调整
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # Use anti-aliased downsampling / 使用抗锯齿下采样
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        # Store the resized mask input in the current frame's mask input dictionary
        # 将调整后的掩码输入存入当前帧的掩码输入字典中
        mask_inputs_per_frame[frame_idx] = mask_inputs
        # Remove the point input for the current frame from the point input dictionary
        # 从点输入字典中移除当前帧的点输入
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame has not been tracked before, treat it as an initial condition frame,
        # meaning the input points will be used to generate segmentation on this frame without using memories from other frames, as in SAM.
        # 如果该帧之前没有被追踪过，我们将其视为初始条件帧，
        # 这意味着输入的点将用于在该帧上生成分割，而不使用其他帧的任何记忆，就像在 SAM 中一样。
        # 否则（如果该帧已经被追踪过），输入的点将用于修正已经追踪到的掩码。
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # Check whether tracking should be in reverse time order
        # 判断是否按时间逆序进行追踪
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        # Get the current object output dictionary and temporary output dictionary
        # 获取当前对象的输出字典和临时输出字典
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # If this is an initial condition frame, or if the model treats all frames with clicks/masks as condition frames,
        # add the frame to the condition outputs.
        # 如果该帧是初始条件帧，或者模型将所有接收到点击/掩码的帧视为条件帧，则将该帧添加到条件输出中。
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        # Choose the storage key based on whether it is a condition frame
        # 根据是否是条件帧选择存储键
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Run single-frame inference to generate the current frame's output
        # 运行单帧推理，生成当前帧的输出
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # Process the output slice for a single object / 处理单个对象的输出切片
            frame_idx=frame_idx,
            batch_size=1,  # Process output slice for a single object / 处理单个对象的输出切片
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # # Skip memory encoder when adding clicks or masks. Memory encoder is executed at the start of `propagate_in_video` (after user completes clicks).
            # 添加点击或掩码时跳过内存编码器。我们在 `propagate_in_video` 的开头执行内存编码器（在用户完成点击后）。
            # 这使我们能够在将对象编码到内存之前，强制对所有对象执行非重叠约束。
            run_mem_encoder=False,
        )
        # Add the current output to the temporary output dictionary (to be used as memory later)
        # 将当前输出添加到临时输出字典中（以便将来用作记忆）
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Adjust the output mask to the original video resolution
        # 将输出掩码调整到原始视频分辨率
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )

        # Get the output mask adjusted to the original video resolution
        # 获取调整为原始视频分辨率的输出掩码
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )

        # Return the frame index, object ID list, and video resolution mask
        # 返回帧索引、对象ID列表和视频分辨率的掩码
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Adjust object scores to original video resolution (video_res_masks),
        and apply non-overlapping constraints for the final output.
        将对象分数调整到原始视频分辨率 (video_res_masks)，
        并为最终输出应用非重叠约束。
        """
        device = inference_state["device"]  # Get device information / 获取设备信息
        video_H = inference_state["video_height"]  # Get video height / 获取视频高度
        video_W = inference_state["video_width"]  # Get video width / 获取视频宽度
        any_res_masks = any_res_masks.to(device, non_blocking=True)  # Move the masks to the device, non-blocking / 将掩码转移到设备上，非阻塞方式
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks  # If the mask resolution is already the same as the video, use it directly / 如果掩码分辨率已经和视频一致，则直接使用
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),  # Resize the masks to the video width and height / 将掩码调整到视频的宽度和高度
                mode="bilinear",  # Use bilinear interpolation / 使用双线性插值
                align_corners=False,  # Do not align corners / 不对齐角点
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)  # Apply non-overlapping constraints if needed / 如果需要应用非重叠约束，执行相应操作

        # Move video_res_mask and any_res_masks to storage_device ：
        # 将video_res_mask和any_res_masks移动至storage_device上：
        video_res_masks = video_res_masks.to(inference_state["storage_device"], non_blocking=False)  # non_blocking=False
        any_res_masks = any_res_masks.to(inference_state["storage_device"], non_blocking=False)  # non_blocking=False
        return any_res_masks, video_res_masks  # Return the original masks and the resized masks at video resolution / 返回调整前的掩码和调整到视频分辨率的掩码

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate temporary outputs `temp_output_dict_per_obj` for each object in a frame into a single output for all objects, including the following steps:
        1) Complete missing objects: If they exist in `output_dict_per_obj`, use their contents to fill in; if not, leave them as placeholders.
        2) Optionally, re-run the memory encoder after applying non-overlapping constraints to the object scores.

        `consolidated_out['maskmem_features']` should be on `inference_state['storage_device']` to save resources.

        将每个对象的临时输出 `temp_output_dict_per_obj` 在某一帧上合并为所有对象的单个输出，包含以下步骤：
        1) 补全任何缺失的对象：如果它们在 `output_dict_per_obj` 中存在，
            则使用 `output_dict_per_obj` 中的内容补全；如果在 `output_dict_per_obj` 中不存在，
            则将其留作占位符。
        2) 如果指定，将在对对象分数应用非重叠约束后重新运行记忆编码器。

        其中consolidated_out['maskmem_features']应当位于inference_state['storage_device']上以节省资源
        """
        batch_size = self._get_obj_num(inference_state)  # Get the number of objects / 获取对象数量
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"  # Choose output dictionary key based on condition /根据条件选择输出字典的键
        # Optionally, we allow consolidating temporary outputs at the original video resolution (for a better mask hint editing experience).
        # 可选地，我们允许在原始视频分辨率上合并临时输出（以提供更好的掩码提示编辑体验）。
        if consolidate_at_video_res:
            assert not run_mem_encoder, "Video resolution consolidation cannot run memory encoder / 记忆编码器不能在视频分辨率下运行"
            consolidated_H = inference_state["video_height"]  # Get video height / 获取视频高度
            consolidated_W = inference_state["video_width"]  # Get video width / 获取视频宽度
            consolidated_mask_key = "pred_masks_video_res"  # Set key for consolidated masks / 设置合并掩码的键
        else:
            consolidated_H = consolidated_W = self.image_size // 4  # Set resolution for consolidated masks / 设定合并掩码的分辨率
            consolidated_mask_key = "pred_masks"  # Set key for consolidated masks / 设置合并掩码的键

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc" will be added after applying non-overlapping constraints to the object scores and re-running the memory encoder.
        # Its "pred_masks" is pre-filled with a large negative value (NO_OBJ_SCORE) to represent missing objects.
        # 初始化 `consolidated_out`。它的 "maskmem_features" 和 "maskmem_pos_enc"
        # 将在对对象分数应用非重叠约束后重新运行记忆编码器时添加。
        # 它的 "pred_masks" 使用一个较大的负值 (NO_OBJ_SCORE) 预填充，以表示缺失的对象。
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,  # Use NO_OBJ_SCORE to fill missing objects / 使用 NO_OBJ_SCORE 作为缺失对象的填充值
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,  # Use NO_OBJ_SCORE to fill missing objects / 使用 NO_OBJ_SCORE 作为缺失对象的填充值
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # 默认将 object_score_logits 设置为 10.0，即假设物体存在，因为 sigmoid(10)=1，这与 `MaskDecoder` 的 `predict_masks` 中的设置相同。
                fill_value=10.0,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]  # Get the temporary output dictionary for each object / 获取每个对象的临时输出字典
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]  # Get the output dictionary for each object / 获取每个对象的输出字典
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)  # Get the temporary output for the corresponding frame / 获取对应帧的临时输出
            # If the object is not in "temp_output_dict_per_obj", fall back and find its previous output in
            # "output_dict_per_obj". We look for "cond_frame_outputs" and "non_cond_frame_outputs" to find previous outputs.
            # 如果该对象未出现在 "temp_output_dict_per_obj" 中，则回退并查找它在
            # "output_dict_per_obj" 中的先前输出。
            # 我们查找 "cond_frame_outputs" 和 "non_cond_frame_outputs" 来为这个对象找到先前的输出。
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object is not found in "output_dict_per_obj", skip it,
            # leave its mask score as the default score (i.e., the placeholder NO_OBJ_SCORE),
            # and set its object pointer to a virtual pointer.
            # 如果该对象也未出现在 "output_dict_per_obj" 中，我们跳过它
            # 并将其掩码分数留作默认分数（即上面 NO_OBJ_SCORE 的占位符），
            # 并将其对象指针设置为虚拟指针。
            if out is None:
                # Fill virtual object pointers for those with no input or tracking results in this frame
                # (only when `run_mem_encoder=True`, i.e., when we need to build memory for tracking).
                # 为那些在这一帧没有任何输入或跟踪结果的对象填充虚拟对象指针
                # （仅在 `run_mem_encoder=True` 时进行，即当我们需要为跟踪构建记忆时）。
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    # Fill object pointers with a virtual pointer based on empty masks
                    # 使用基于空掩码的虚拟指针填充对象指针
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add temporary object output masks to the consolidated output masks
            # 将临时对象输出掩码添加到合并输出掩码中
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask  # If the mask resolution matches, directly add it / 如果掩码分辨率匹配，直接添加
            else:
                # If the temporary object mask resolution differs, resize it first
                # 如果临时对象掩码的分辨率不同，首先调整大小
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],  # Resize to the consolidated mask resolution / 调整到合并掩码的分辨率
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]  # Update object pointers / 更新对象指针
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        # Optionally, apply non-overlapping constraints on the consolidated scores and re-run the memory encoder
        # 可选地，在合并的分数上应用非重叠约束，并重新运行记忆编码器
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,  # These frames are the result of user interaction / 这些帧是用户交互的结果
            )

            # Update memory encoder features, already on storage_devic / 更新记忆编码器特征,已经位于storage_device上
            consolidated_out["maskmem_features"] = maskmem_features
            # Update memory encoder positional encoding / 更新记忆编码器位置编码
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out  # Return the consolidated output / 返回合并后的输出

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """
        Get a virtual object pointer based on the empty mask from the current frame.
        根据当前帧上的空掩码获取一个虚拟对象指针。
        """
        # Create a virtual (empty) mask with only one object
        # 创建一个只有一个对象的虚拟（空）掩码
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),  # The mask size matches the image size / 掩码尺寸与图像尺寸一致
            dtype=torch.float32,
            device=inference_state["device"],  # Use the device from the inference state / 使用推理状态中的设备
        )

        # Get the current image features / 获取当前图像特征
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Input the empty mask and above image features to get a virtual object pointer
        # 将空掩码和上述图像特征输入，获取一个虚拟对象指针
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,  # Indicates this is the initial condition frame / 表示这是初始条件帧
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,  # No point inputs / 没有点输入
            mask_inputs=mask_inputs,  # Use the empty mask / 使用空掩码
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,  # Do not run memory encoder / 不运行记忆编码器
            prev_sam_mask_logits=None,
            preloading_memory_cond_frame_idx=None, # Do not pass in the preloading memory condition frame index / 不传入预加载内存库中条件帧索引
        )
        return current_out["obj_ptr"]  # Return the virtual object pointer / 返回虚拟对象指针

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """
        Prepare the inference state and merge temporary outputs before tracking.
        准备推理状态，并在跟踪之前合并临时输出。
        """
        # Tracking has started, new objects cannot be added before session reset.
        # 跟踪已经开始，不允许在会话重置之前添加新对象。
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)  # Get the number of objects (items to be tracked) / 获取对象(需要跟踪的物体)的数量
        # print(f"propagate_in_video_preflight中batch_size:{batch_size}")

        # Merge the temporary outputs of each object from "temp_output_dict_per_obj" and add them to "output_dict".
        # 合并 "temp_output_dict_per_obj" 中的每个对象的临时输出，并将其添加到 "output_dict" 中。
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains the indices of frames that have merged temporary outputs (whether in the current call or previous calls to `propagate_in_video_preflight`).
        # "consolidated_frame_inds" 包含那些已合并临时输出的帧的索引（无论是在当前调用还是之前调用 `propagate_in_video_preflight` 时）。
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]

        for is_cond in [False, True]:
            # Merge conditional and non-conditional temporary outputs separately
            # 分别合并条件和非条件临时输出
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all frames that contain temporary outputs for any object
            # 查找包含任何对象的临时输出的所有帧
            # （这些应是刚刚通过 `add_new_points_or_box` 或 `add_new_mask` 收到点击以获取掩码输入的帧）
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            # Collect and update the temporary frame inference result indices for each object, and merge them into the global frame index record
            # 收集并更新每个对象的临时帧推理结果的索引，并将这些索引合并到全局的帧索引记录中
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # print(f"{'条件帧' if is_cond else '非条件帧'}合并的帧索引: {sorted(temp_frame_inds)}")
            # # 非条件帧[]，条件帧[90,105]

            # Merge the temporary outputs of all objects on these frames
            # 合并这些帧上的所有对象的临时输出
            for frame_idx in temp_frame_inds:
                # print(f"frame_idx:{frame_idx}")
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )

                # def print_consolidated_out(consolidated_out):  # 打印合并的输出
                #     for key, value in consolidated_out.items():
                #         if isinstance(value, torch.Tensor):
                #             print(f"Key: {key}, Value: Tensor, Shape: {value.shape}")
                #         elif isinstance(value, dict):
                #             print(f"Key: {key}, Value: dict")
                #             print_consolidated_out(value)  # 递归打印嵌套的字典
                #         elif isinstance(value, list):
                #             print(f"Key: {key}, Value: list, Length: {len(value)}")
                #         else:
                #             print(f"Key: {key}, Value: {value}")
                # print_consolidated_out(consolidated_out)

                # Merge them into "output_dict" and create slices for each object
                # 将它们合并到 "output_dict" 中，同时创建每个对象的切片
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # Clear non-conditional memory around the input frames
                    # 清除周围帧的非条件记忆
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # Clear the temporary outputs in `temp_output_dict_per_obj`
            # 清除 `temp_output_dict_per_obj` 中的临时输出
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # Edge case: If outputs were added to "cond_frame_outputs", remove any previous "non_cond_frame_outputs" on the same frame
        # (We don't want both conditional and non-conditional outputs on the same frame).
        # 边界情况：如果在 "cond_frame_outputs" 中添加了输出，则删除在同一帧上的任何先前的 "non_cond_frame_outputs"
        # （我们不希望在同一帧上即使条件帧又是非条件帧）。
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Ensure that the frame indices in "consolidated_frame_inds" are exactly those frames with point inputs or mask inputs (which should be true in the correct workflow).
        # 确保 "consolidated_frame_inds" 中的帧索引正好是那些有点输入或掩码输入的帧（在正确的工作流下应为真）。
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())

        # assert all_consolidated_frame_inds == input_frames_inds  # 确保所有合并的帧索引都与输入帧索引一致
        # I tried implementing release_old_frames() to remove old conditional and non-conditional frames to save memory, but was blocked by this assertion.
        # I personally think it's fine to comment out this assertion, as releasing old conditional frames will inevitably clear the consolidated_frame_inds index,
        # and in fact, these indices will no longer be used in future propagation.
        # 我尝试实现release_old_frames()以去除旧的条件帧与非条件帧以节省内存时,被官方的这个断言卡住,
        # 注释掉这个断言,我个人认为是没有问题的,释放旧条件帧势必会清空consolidated_frame_inds的索引,事实上,这些索引在此后的propagate传播中都不会再被使用到

        # assert all_consolidated_frame_inds == input_frames_inds  # 确保所有合并的帧索引都与输入帧索引一致

    def print_gpu_memory(self):  # TODO:Temporarily used for checking GPU memory usage during development; will not be used in production.
        try:
            # 使用 nvidia-smi 获取显存信息 / Use nvidia-smi to get GPU memory information
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"])
            result = result.decode("utf-8").strip().split("\n")
            # 每个 GPU 的显存使用情况 (used, free) / Memory usage for each GPU (used, free)
            gpu_memory = [tuple(map(int, line.split(", "))) for line in result]
            if gpu_memory:
                for idx, (used, free) in enumerate(gpu_memory):
                    print(f"GPU{idx} memory - use: {used} MB, free: {free} MB")
        except Exception as e:
            print(f"Error in getting GPU memory: {e}")
            return None

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """
        Propagate input points for tracking throughout the video.
        在整个视频中传播输入点以进行跟踪。
        """
        # Prepare the inference state and merge temporary outputs before tracking
        # 在跟踪之前准备推理状态并合并临时输出
        self.propagate_in_video_preflight(inference_state)  # Merge temporary outputs, clear unnecessary memory, and ensure consistency in the inference state / 合并临时输出，清除无用的记忆，并确保推理状态的一致性

        # Extract information from the processed inference_state / 从处理好的 inference_state 中提取信息
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # print(f"从处理好的 inference_state 中提取的信息：")
        # cond_frame_outputs_count = len(output_dict.get("cond_frame_outputs", {}))
        # non_cond_frame_outputs_count = len(output_dict.get("non_cond_frame_outputs", {}))
        # print(f"output_dict中 条件帧：{cond_frame_outputs_count}，非条件帧：{non_cond_frame_outputs_count}")

        # # output_dict中存在此前所有推理序列和当前推理序列的条件帧，只存在此前所有推理序列、不存在当前推理序列的非条件帧
        # consolidated_cond_frame_count = len(consolidated_frame_inds["cond_frame_outputs"])
        # consolidated_non_cond_frame_count = len(consolidated_frame_inds["non_cond_frame_outputs"])
        # print(f"consolidated_frame_inds中 条件帧：{consolidated_cond_frame_count}，非条件帧：{consolidated_non_cond_frame_count}")

        # print(f"obj_ids追踪的对象 ID 列表: {obj_ids}")
        # print(f"视频当前总帧数num_frames: {num_frames}")

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points provided; please add points first / 没有提供点；请先添加点")

        # Decide whether to clear non-conditional memory based on settings
        # 根据设置决定是否清除非条件记忆
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )
        # print(f"是否清除非条件记忆clear_non_cond_mem：{clear_non_cond_mem}")

        # Set start index, end index, and processing order
        # 设置起始索引、结束索引和处理顺序
        if start_frame_idx is None:
            # Default to starting from the first frame with input points / 默认从第一个有输入点的帧开始
            start_frame_idx = min(output_dict["cond_frame_outputs"])
            # print(f"起始帧(从第一个条件帧开始):{start_frame_idx}")
        if max_frame_num_to_track is None:
            # print(f"max_frame_num_to_track is None,跟踪视频所有帧")
            # Default to tracking all frames in the video / 默认跟踪视频中的所有帧
            max_frame_num_to_track = num_frames
        if reverse:
            # If tracking in reverse, calculate the end frame index / 如果是倒序跟踪，计算结束帧的索引
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track + 1, 0)  # 这个+1是我们自己加的，保证传播长度准确
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # Skip reverse tracking if starting from frame 0 / 如果从第0帧开始，跳过倒序跟踪
        else:
            # If tracking in forward direction, calculate the end frame index
            # 如果是正序跟踪，计算结束帧的索引
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        # print(f"结束帧(由起始帧、跟踪方向和最大跟踪长度决定)：{end_frame_idx}")

        # Iterate over each frame in the processing order / 遍历处理顺序中的每一帧
        for frame_idx in tqdm(processing_order, desc=f"propagate in video start:{start_frame_idx},end:{end_frame_idx}"):
            # Skip frames that have already been processed (those in consolidated output)
            # 跳过那些已经在合并输出中的帧（这些帧已经接收到输入点击或掩码）。
            # 注意，我们不能直接执行批处理推理，因为每个对象上的点击数可能不同。
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                # print(f"帧索引{frame_idx}，已有条件帧输出，不对其推理")
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # Clear non-conditional memory around the input frame / 清除周围帧的非条件记忆
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                # print(f"帧索引{frame_idx}，已有非条件帧输出，不对其推理")
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                # print(f"帧索引{frame_idx}，未经过处理，进行单帧推理！")
                # For unprocessed frames, perform single-frame inference
                # 对于未处理的帧，进行单帧推理
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )

                output_dict[storage_key][frame_idx] = current_out  # Store current frame's output / current_out中['maskmem_features']和['pred_masks']位于storage_device上

            # Create output slices for each object for later interaction
            # 为每个对象创建输出切片，以便后续交互
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )  # 某一帧上的推理结果按对象进行分割并添加到inference_state中

            # Record whether a frame has been tracked and store its tracking direction / 记录某帧是否已经被跟踪过，并且保存该帧的跟踪方向
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Adjust output masks to the original video resolution / 将输出掩码调整到原始视频分辨率（直接使用GPU上的掩码分数以避免中间的CPU转换）
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )

            yield frame_idx, obj_ids, video_res_masks  # Yield the current frame index, object IDs, and video resolution masks / 返回当前帧索引、对象ID和视频分辨率掩码

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Splits the multi-object output into individual object slices and adds them to `output_dict_per_obj`.
        The resulting slices share the same tensor storage.
        将多对象输出拆分为每个对象的输出切片，并将它们添加到 `output_dict_per_obj` 中。
        结果切片共享相同的张量存储。
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            # 为每个对象创建切片
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            # If maskmem_features exists, add it to the object output
            # 如果有 maskmem_features，则将其添加到对象输出中
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]  # On the storage_device / 在storage_device上
            # If maskmem_pos_enc exists, add it to the object output
            # 如果有 maskmem_pos_enc，则将其添加到对象输出中
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]   # # On the GPU
            # Add the object output to `output_dict_per_obj`
            # 将对象的输出添加到 `output_dict_per_obj` 中
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
            self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """
        Clears all input points or masks for the given object in the specified frame.
        在指定帧上为给定的物体清除所有输入点或掩码
        """
        # Get the object index based on the object ID
        # 根据对象 ID 获取对象索引
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear all condition information for the specified frame
        # 清除该帧上的所有条件信息
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check if there are still any inputs for objects in the current frame
        # 检查当前帧是否仍然有任何对象的输入
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If there are no more inputs for the current frame, further clear its condition state
        # 如果该帧不再有任何对象的输入，则进一步清除该帧的条件状态
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

            # Remove the conditional output for this frame (may downgrade to non-conditional frame output)
            # 移除该帧的条件输出（可能降级为非条件帧输出）
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # Since the frame no longer receives inputs, it is no longer a conditional frame, downgrade it to non-conditional output
                # 由于该帧不再接收输入，它不再是条件帧，将其降级为非条件帧输出
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)

            # Similarly, process the sliced outputs for each object
            # 同样的处理每个对象的切片输出
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # If all conditional frames are removed, reset the tracking outputs
            # 如果所有条件帧都被移除，也要清除跟踪输出
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return

        # Finally, output the updated masks for each object (after removing the inputs above)
        # 最后，输出每个对象的更新后的掩码（在上面移除输入后）
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """
        Reset all input points or masks across all frames in the video.
        删除视频中所有帧的所有输入点或掩码。
        """
        # Reset tracking results / 重置跟踪结果
        self._reset_tracking_results(inference_state)
        # Reset tracking results / 重置跟踪结果
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """
        Reset all tracking inputs and results in the video.
        重置视频中的所有跟踪输入和结果。
        """
        # Clear point inputs for each object / 清空每个对象的点输入
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        # Clear mask inputs for each object / 清空每个对象的掩码输入
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        # Clear condition and non-condition outputs in output_dict_per_obj for each object / 清空每个对象的输出字典中的条件帧输出和非条件帧输出
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        # Clear condition and non-condition outputs in temp_output_dict_per_obj for each object / 清空每个对象的临时输出字典中的条件帧输出和非条件帧输出
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        # Clear condition and non-condition outputs in output_dict for each object / 清空总输出字典中的条件帧输出和非条件帧输出
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        # Clear consolidated frame indices / 清空合并帧索引
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        # Reset tracking state / 重置跟踪状态
        inference_state["tracking_has_started"] = False
        # Clear already tracked frames / 清空已经跟踪的帧
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """
        Compute image features for a given frame.
        计算给定帧上的图像特征。
        """
        # Look up in cache first / 首先在缓存中查找
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss - perform inference on a single image / 缓存丢失 -- 我们将对单张图像进行推理
            device = inference_state["device"]
            # Retrieve the current frame image from the cache, move it to the device, convert to float, and add a batch dimension
            # 从缓存中获取当前帧的图像，转移到设备并进行浮点数转换，增加一个批次维度
            target_idx = inference_state["images_idx"].index(frame_idx) # images tensor不连续记录视频帧，需要通过images_idx映射列表才能知道images中什么位置对于真实第几帧
            # print(f"希望获取{frame_idx}实际帧,images_idx中的位置{target_idx}")
            image = inference_state["images"][target_idx].to(device).float().unsqueeze(0)
            # Perform forward pass to compute image features / 通过前向传播计算图像特征
            backbone_out = self.forward_image(image)
            # Cache the features of the most recent frame for repeated interactions (can use LRU cache for more frames in the future)
            # 缓存最近一帧的特征（以便对同一帧进行重复交互；将来可以使用LRU缓存处理更多帧）
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # Expand features to match the number of objects
        # 将特征扩展到与对象数量相同的维度
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        # Expand each feature map to match batch size
        # 扩展每个特征图以匹配批次大小
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        # Prepare backbone features / 准备骨干网的特征
        features = self._prepare_backbone_features(expanded_backbone_out)
        # Return a tuple containing the expanded image and prepared features / 返回包含扩展图像和准备好的特征的元组
        features = (expanded_image,) + features
        return features

    # 释放旧的帧，以节约显存和内存
    def release_old_frames(self, inference_state, frame_idx, max_inference_state_frames, pre_frames, release_images=False):
        '''
        Clear frames that will no longer be used for inference. Typically, max_inference_state_frames is greater than the propagation length.
        :param inference_state: Inference memory store
        :param frame_idx: Current frame index
        :param max_inference_state_frames: Maximum number of frames to retain
        :param pre_frames: Number of frames in the preloaded memory store. Ensure frames within the preloaded memory store are not cleared. (pre_frames-1) is the maximum frame index for the preloaded memory store.
        :param release_images: Whether to clear old video frames

        清除确保不会再进行推理的帧，一般max_inference_state_frames大于传播长度
        :param
        inference_state: 推理内存库
        frame_idx: 当前帧索引
        max_inference_state_frames: 保留的最大帧数
        pre_frames: 预加载内存库的帧数，需要保证不清除预加载内存库的帧数。(pre_frames-1)为预加载内存库的最大帧索引
        vis_frame_stride: 如果为-1说明不进行可视化，则可以清除旧的视频帧
        '''
        # print(f"当前帧{frame_idx},保留最大帧数{max_inference_state_frames}")

        # Set the oldest allowed frame index to `frame_idx - max_inference_state_frames`, retaining only the most recent max_inference_state_frames frames
        # 设置允许保留的最旧帧索引为 `frame_idx - max_inference_state_frames`，即只保留最近的 max_inference_state_frames 帧
        oldest_allowed_idx = frame_idx - max_inference_state_frames

        # Retrieve all frame indices stored in inference_state['output_dict']
        # 获取所有存储在inference_state['output_dict']中的帧索引
        all_cond_frames_idx = inference_state['output_dict']['cond_frame_outputs'].keys()
        all_non_cond_frames_idx = inference_state['output_dict']['non_cond_frame_outputs'].keys()
        # Frame indices older than oldest_allowed_idx but greater than (pre_frames - 1)
        old_cond_frames_idx = [idx for idx in all_cond_frames_idx if (pre_frames - 1) < idx <= oldest_allowed_idx]  # 小于oldest_allowed_idx且大于预加载内存库（pre_frames-1）的帧索引
        # Same condition for non-condition frames
        old_non_cond_frames_idx = [idx for idx in all_non_cond_frames_idx if (pre_frames - 1) < idx <= oldest_allowed_idx]  # 小于oldest_allowed_idx且大于预加载内存库（pre_frames-1）的帧索引
        # print(f"old_cond_frames_idx:{old_cond_frames_idx}")
        # print(f"old_non_cond_frames_idx:{old_non_cond_frames_idx}")

        for old_idx in old_non_cond_frames_idx:
            # Remove old non-condition frames from 'output_dict' / 删除'output_dict'中旧的非条件帧
            inference_state['output_dict']['non_cond_frame_outputs'].pop(old_idx,None)
            # Remove old non-condition frames from 'output_dict_per_obj' / 删除'output_dict_per_obj'中旧的非条件帧
            for obj in inference_state['output_dict_per_obj'].keys():
                inference_state['output_dict_per_obj'][obj]['non_cond_frame_outputs'].pop(old_idx,None)

        for old_idx in old_cond_frames_idx:
            # Remove old condition frames from 'output_dict' and 'consolidated_frame_inds' / 同时删除'output_dict'和'consolidated_frame_inds'中旧的条件帧
            inference_state['output_dict']['cond_frame_outputs'].pop(old_idx,None)
            inference_state['consolidated_frame_inds']['cond_frame_outputs'].discard(old_idx)
            # Remove old condition frames from 'output_dict_per_obj' / 删除'output_dict_per_obj'中旧的条件帧
            for obj in inference_state['output_dict_per_obj'].keys():
                inference_state['output_dict_per_obj'][obj]['cond_frame_outputs'].pop(old_idx,None)

        if release_images: # Clear old video frames / 清除旧的视频帧
            old_image_indices = [idx for idx in inference_state["images_idx"] if (pre_frames - 1) < idx <= oldest_allowed_idx]
            # print(f"需要删除的图像帧信息：{old_image_indices}")
            image_idx_to_remove = []
            for old_idx in old_image_indices:
                # Convert actual video frame index to index in 'images'
                old_image_idx = inference_state["images_idx"].index(old_idx)  # 需要被删除的真实视频帧索引转化为images中对应索引
                image_idx_to_remove.append(old_image_idx)
            # print(f"需要删除的图像帧信息对应images索引：{image_idx_to_remove}")

            # Perform deletion on both 'images' and 'images_idx'
            # Use torch.index_select to retain only necessary frames
            # 对 images 和 images_idx 都进行删除操作
            # 使用torch.index_select来删除不需要的帧，保留其他帧
            mask = torch.tensor([i for i in range(inference_state["images"].size(0)) if i not in image_idx_to_remove])
            inference_state["images"] = torch.index_select(inference_state["images"], dim=0, index=mask)
            inference_state["images_idx"] = [idx for idx in inference_state["images_idx"] if idx not in old_image_indices]
            # print(f"删除后的 images_idx: {inference_state['images_idx']}")
            # print(f"删除后images长度：{len(inference_state['images'])}")

            assert len(inference_state["images"]) == len(inference_state["images_idx"])  # 确保images和images_idx长度一致

        # Perform garbage collection after batch deletion / 批量删除后主动调用垃圾回收
        gc.collect()

        # print(f"output_dict条件帧索引：{inference_state['output_dict']['cond_frame_outputs'].keys()}")
        # print(f"output_dict非条件帧索引：{inference_state['output_dict']['non_cond_frame_outputs'].keys()}")
        # print(f"consolidated_frame_inds条件帧索引：{inference_state['consolidated_frame_inds']['cond_frame_outputs']}")

    # Run inference on a single frame / 执行单帧推理
    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """
        Run tracking on a single frame based on current input and previous memory.
        在单帧上运行跟踪，基于当前输入和之前的内存。
        """

        # Get image features for the current frame / 获取当前帧的图像特征
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Ensure point inputs and mask inputs do not appear on the same frame / 确保点输入和掩膜输入在同一帧上不会同时出现
        assert point_inputs is None or mask_inputs is None

        # 打印track_step的输入
        # print(f"is_init_cond_frame: {is_init_cond_frame}")
        # print(f"current_vision_feats: {current_vision_feats[0].shape}")  # torch.Size([65536, 1, 32]
        # print(f"current_vision_pos_embeds: {current_vision_pos_embeds[0].shape}")  # torch.Size([65536, 1, 256])
        # print(f"feat_sizes: {feat_sizes}")  # [(256, 256), (128, 128), (64, 64)]
        # print(f"track_in_reverse: {reverse}")
        # print(f"prev_sam_mask_logits: {prev_sam_mask_logits}")  # None

        # Run tracking step with current frame features and inputs
        # 运行跟踪步骤，传入当前帧的特征和输入
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            # Pass preloading memory condition frame index for memory attention
            preloading_memory_cond_frame_idx=inference_state["preloading_memory_cond_frame_idx"], # 传入预加载内存库中条件帧索引,用于保证其参与到memory attention计算中
        )

        # Optionally move output to CPU memory to save GPU space
        # 可选：将输出转移到CPU内存中以节省GPU空间
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            # Convert features to bfloat16 to save memory
            # 将特征转换为bfloat16以节省内存
            maskmem_features = maskmem_features.to(torch.bfloat16)
            # Move features to storage device (e.g., CPU)
            # 将特征转移到存储设备（如CPU）中
            maskmem_features = maskmem_features.to(storage_device, non_blocking=False)  # non_blocking=False

        pred_masks_gpu = current_out["pred_masks"]
        # Fill holes in predicted masks if necessary / 如果需要，填补预测掩膜中的空洞
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        # Move predicted masks to storage device / 将预测掩膜转移到存储设备中
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=False)  # non_blocking=False

        # "maskmem_pos_enc" is the same across all frames, so only one copy is stored
        # "maskmem_pos_enc"在所有帧中是相同的，所以只需要存储一份副本
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # Object pointers are small tensors, so they are always kept on GPU memory for fast access
        # 对象指针是一个小张量，所以始终保留在GPU内存中以便快速访问
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]

        # Create a compact version of the current frame's output to reduce state size
        # 制作当前帧输出的紧凑版本，以减少状态大小
        compact_current_out = {
            "maskmem_features": maskmem_features,  # on storage_device
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,  # on storage_device
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }

        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts
    ):
        """
        Run memory encoder on `high_res_masks`, typically after applying non-overlap constraints to object scores.
        Recalculate memory through the memory encoder as their scores have changed.
        在 `high_res_masks` 上运行记忆编码器。通常是在对对象分数应用非重叠约束之后进行的。
        由于它们的分数发生了变化，所以它们的记忆也需要通过记忆编码器重新计算。
        """
        # Get image features for the current frame / 获取当前帧的图像特征
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        # Encode new memory / 对新记忆进行编码
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # Optionally move output to CPU memory to save GPU space / 可选：将输出转移到CPU内存中以节省GPU空间
        storage_device = inference_state["storage_device"]
        # Convert memory features to bfloat16 to save memory / 将记忆特征转换为bfloat16以节省内存
        maskmem_features = maskmem_features.to(torch.bfloat16)
        # Move features to storage device / 将特征转移到存储设备中
        maskmem_features = maskmem_features.to(storage_device, non_blocking=False)  # non_blocking=False
        # print("maskmem_features:", maskmem_features.device)  # CPU
        # "maskmem_pos_enc" is the same across all frames, so only one copy is stored
        # "maskmem_pos_enc" 在所有帧中是相同的，所以只需存储一份副本
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is identical across all frames and objects, so we cache it as
        a constant to reduce session storage size.
        `maskmem_pos_enc` 在所有帧和对象中是相同的，因此我们将其缓存为常量以减少会话存储大小。
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be a list of tensors or None
        # "out_maskmem_pos_enc" 应该是张量的列表或 None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]

        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                # Ensure the output is a list of tensors / 确保输出是一个张量的列表
                assert isinstance(out_maskmem_pos_enc, list)

                # Slice only one object since it is the same for all objects
                # 只取一个对象的切片，因为它在所有对象中都是相同的
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                # Cache maskmem_pos_enc / 缓存 maskmem_pos_enc
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                # Use cached maskmem_pos_enc / 使用缓存的 maskmem_pos_enc
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]

            # Expand the cached maskmem_pos_enc to the actual batch size
            # 将缓存的 maskmem_pos_enc 扩展到实际批量大小
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Removes an object ID from the tracking state. If `strict` is True, checks if
        the object ID exists and raises an error if it doesn't.
        从跟踪状态中移除一个对象ID。如果strict为True，则检查对象ID是否存在，
        如果不存在则抛出错误。
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []

        # Check if the object ID to remove exists and raise an error based on `strict`.
        # 检查是否要移除的对象ID存在，并根据strict设置可能抛出错误。
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object ID {obj_id} as it doesn't exist. "
                f"Existing object IDs: {inference_state['obj_ids']}. / "
                f"无法移除对象ID {obj_id}，因为它不存在。"
                f"所有现有的对象ID：{inference_state['obj_ids']}。"
            )

        # If this is the only remaining object ID, reset the state directly.
        # 如果这是唯一剩下的对象ID，直接重置状态。
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # If there are other objects left after removing this one, delete its storage
        # from the inference state tensors.
        # 在移除这个对象ID后还有其他剩余对象。在这种情况下，
        # 我们需要从推理状态张量中删除对象的存储。

        # Step 0: Clear inputs for this object on frames with point or mask inputs
        # (This step is necessary as it may downgrade conditional frames to non-conditional frames).
        # 第0步：清除这个对象在有点或掩码输入的帧上的输入
        # （注意这一步是必须的，因为它可能将条件帧降级为非条件帧）。
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update object ID mappings (This step must be done after Step 0
        # as Step 0 still requires the old object ID mapping in the inference state).
        # 第1步：更新对象ID映射（注意这一步必须在第0步之后执行，
        # 因为第0步仍然需要使用推理状态中的旧对象ID映射）。
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))

        # Build new mappings 构建新的映射
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: Update tensor storage for each object by remapping indices.
        # (Note that "consolidated_frame_inds" doesn't need to be updated in this step
        # as it was already handled in Step 0).
        # 第2步：对于每个对象的张量存储，我们将它们的对象索引移动到字典键中。
        # （注意"consolidated_frame_inds"不需要在此步骤中更新，因为它已经在第0步中处理了）。
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # Step 3: Slice tensor storage for remaining object IDs and reconstruct slices for each object.
        # 第3步：对于打包的张量存储，我们索引剩余的ID并重建每个对象的切片。
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [
                    x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]
                ]
                # "maskmem_pos_enc" is identical across frames, so only one copy is stored.
                # "maskmem_pos_enc"在各帧间相同，因此我们只需要存储一份副本
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][
                    remain_old_obj_inds
                ]
                # Update slices for each object. / 还需要更新每个对象的切片
                self._add_output_per_object(
                    inference_state, frame_idx, out, storage_key
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        # Step 4: Collect outputs for frames in `obj_input_frames_inds`,
        # which may show updated masks for objects occluded by the removed object.
        # 第4步：进一步收集`obj_input_frames_inds`中那些帧上的输出，
        # 这些帧可能会显示被移除对象遮挡的对象的更新掩码。
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Removes non-conditional memory around an input frame. When the user provides
        correction clicks, non-conditional memory in surrounding frames may still contain
        outdated object appearance information, which could confuse the model.

        This method clears non-conditional memory in surrounding frames to prevent the
        model from receiving both old and new information about an object.

        移除输入帧周围的非条件记忆。当用户提供修正点击时，周围帧的非条件记忆
        可能仍然包含过时的对象外观信息，可能会混淆模型。

        该方法清除周围帧的非条件记忆，以避免给模型提供关于对象的旧信息和新信息。
        """
        r = self.memory_temporal_stride_for_eval
        # Compute the start and end indices of frames to clear.
        # 计算需要清除的帧的起始索引和结束索引
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]

        for t in range(frame_idx_begin, frame_idx_end + 1):
            # Remove non-conditional outputs for the specified frame from the output dictionary.
            # 从输出字典中移除指定帧的非条件输出
            non_cond_frame_outputs.pop(t, None)

            # Remove non-conditional outputs for the specified frame from each object's output dictionary.
            # 从每个对象的输出字典中移除指定帧的非条件输出
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)
