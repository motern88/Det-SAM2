# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

import subprocess  # TODO;暂时开发用于查看内存占用，后续删除

# 一个大负值，作为缺少物体时的占位符得分
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,  # 图像编码器
        memory_attention,  # 用于记忆注意力机制
        memory_encoder,  # 记忆编码器
        num_maskmem=7,  # 默认值为1个输入帧加上6个前面的帧
        image_size=512,  # 图像大小
        backbone_stride=16,  # 图像骨干网络输出的步幅
        sigmoid_scale_for_mem_enc=1.0,  # 掩码Sigmoid概率的缩放因子
        sigmoid_bias_for_mem_enc=0.0,  # 掩码Sigmoid概率的偏置因子
        # 在评估过程中，是否对具有点击的交互帧上的Sigmoid掩码Logits进行二值化处理
        binarize_mask_from_pts_for_mem_enc=False,
        # 在具有掩码输入的帧上，是否直接输出输入掩码而不使用SAM提示编码器+掩码解码器
        use_mask_input_as_output_without_sam=False,
        # 在内存注意力中参与的最大条件帧数（-1表示无限制；如果条件帧数超过此限制，
        # 我们只对编码器中最接近的`max_cond_frames_in_attn`条件帧进行交叉注意力来跟踪每一帧）。
        # 这使模型在处理大量注释帧时具有时间局部性（因为更接近的帧应该更重要），也避免了GPU OOM。
        max_cond_frames_in_attn=20,  # -1
        # 在第一帧上，是否直接将无内存嵌入添加到图像特征中（而不是使用transformer编码器）
        directly_add_no_mem_embed=False,
        # 是否在SAM掩码解码器中使用高分辨率特征图
        use_high_res_features_in_sam=False,
        # 是否在初始条件帧上的第一个点击上输出多个（3个）掩码
        multimask_output_in_sam=False,
        # 使用multimask_output_in_sam的最小和最大点击次数（仅当`multimask_output_in_sam=True`时相关；
        # 默认值为1，表示仅第一个点击提供多掩码输出；另请注意，box计作两个点）
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # 是否在跟踪时也使用多掩码输出（不仅仅是在初始条件帧上的第一个点击上；仅当`multimask_output_in_sam=True`时相关）
        multimask_output_for_tracking=False,
        # 是否为对象指针使用多掩码标记；仅当同时
        # use_obj_ptrs_in_encoder=True和multimask_output_for_tracking=True时相关
        use_multimask_token_for_obj_ptr: bool = False,
        # 是否使用Sigmoid将IoUs预测限制在[0-1]范围内
        iou_prediction_use_sigmoid=False,
        # 在评估期间，内存库的时间步幅（即XMem和Cutie中的`r`参数；XMem和Cutie使用r=5）。
        # 对于r>1，(self.num_maskmem - 1)个非条件内存帧包括
        # (self.num_maskmem - 2)个从每r帧中取出的最近帧，以及最后一帧。
        memory_temporal_stride_for_eval=1,
        # 在评估期间是否在内存编码器中对对象掩码应用非重叠约束（以避免/缓解掩码重叠）
        non_overlap_masks_for_mem_enc=False,
        # 是否在编码器中交叉注意来自其他帧的对象指针（基于SAM输出标记）
        use_obj_ptrs_in_encoder=False,
        # 编码器交叉注意的最大对象指针数（仅当`use_obj_ptrs_in_encoder=True`时相关）
        max_obj_ptrs_in_encoder=16,
        # 是否在编码器中的对象指针上添加时间位置编码（仅当`use_obj_ptrs_in_encoder=True`时相关）
        add_tpos_enc_to_obj_ptrs=True,
        # 是否为对象指针中的时间位置编码添加额外的线性投影层，以避免潜在的干扰
        # 空间位置编码（仅当同时`use_obj_ptrs_in_encoder=True`和`add_tpos_enc_to_obj_ptrs=True`时相关）
        proj_tpos_enc_in_obj_ptrs=False,
        # 是否在对象指针中的时间位置编码使用有符号距离（而不是无符号绝对距离）
        # （仅在use_obj_ptrs_in_encoder = True和add_tpos_enc_to_obj_ptrs = True两者都为真时相关）
        use_signed_tpos_enc_to_obj_ptrs=False,
        # 在评估期间，是否仅在编码器中关注过去的对象指针（当前帧之前），这可能有助于避免未来信息干扰初始跟踪（仅在启用编码器中的对象指针时相关）。
        only_obj_ptrs_in_the_past_for_eval=False,
        # 是否预测帧中是否存在对象
        pred_obj_scores: bool = False,
        # 是否使用MLP预测对象得分
        pred_obj_scores_mlp: bool = False,
        # 仅在pred_obj_scores=True和use_obj_ptrs_in_encoder=True时相关；
        # 在没有对象时，是否使用固定的无对象指针，
        # 或将其用作解码器生成的对象指针的附加嵌入
        fixed_no_obj_ptr: bool = False,
        # 软无对象，即以软方式混合无对象指针，
        # 希望能够更容易地恢复错误，并减轻错误的积累
        soft_no_obj_ptr: bool = False,
        # 是否使用MLP进行对象指针投影
        use_mlp_for_obj_ptr_proj: bool = False,
        # 为特殊帧添加无物体嵌入
        no_obj_embed_spatial: bool = False,
        # 构建SAM掩码解码器的额外参数；如果不为None，它应该是一个传递给`MaskDecoder`类的kwargs字典。
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,  # 是否编译图像编码器
    ):
        super().__init__()

        # 部分1：图像骨干网络
        self.image_encoder = image_encoder
        # 对于高分辨率设置，使用0、1、2层，或者仅使用默认设置的第2层
        self.use_high_res_features_in_sam = use_high_res_features_in_sam  # 是否使用高分辨率特征
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1  # 特征层的数量
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder  # 在编码器中使用对象指针
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder  # 编码器中最大对象指针数量
        if use_obj_ptrs_in_encoder:
            # 一个卷积层将掩码提示下采样到步幅4（与低分辨率SAM掩码Logits相同的步幅）
            # 并将其缩放从0~1更改为SAM Logits缩放，使其可以输入到SAM掩码解码器生成指针。
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)  # 掩码下采样
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs  # 是否添加时间位置编码到对象指针
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # 这些选项需要一起使用
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs  # 在对象指针中投影时间位置编码
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval  # 在评估时仅使用过去的对象指针

        # 部分2：用于条件当前帧的视觉特征与过去帧的记忆（和对象指针）的记忆注意力
        self.memory_attention = memory_attention  # 记忆注意力模块
        self.hidden_dim = memory_attention.d_model  # 隐藏维度

        # 部分3：用于前一帧输出的记忆编码器
        self.memory_encoder = memory_encoder  # 记忆编码器
        self.mem_dim = self.hidden_dim  # 记忆维度
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # 如果沿通道维度有记忆压缩
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]  # 设置记忆维度
        self.num_maskmem = num_maskmem  # 可访问的记忆数量
        # 记忆的时间编码
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )  # 掩码记忆时间位置编码
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)  # 截断正态初始化
        # 一个单一的token，用于表示前一帧没有记忆嵌入
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))  # 无记忆嵌入的标志
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))  # 无记忆位置编码的标志
        trunc_normal_(self.no_mem_embed, std=0.02)  # 截断正态初始化
        trunc_normal_(self.no_mem_pos_enc, std=0.02)  # 截断正态初始化
        self.directly_add_no_mem_embed = directly_add_no_mem_embed  # 是否直接添加无记忆嵌入
        # 对输出的原始掩码Logits应用Sigmoid（将范围从(-inf, +inf)转换为(0, 1)）， 然后再输入到记忆编码器
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc  # 记忆编码器的Sigmoid缩放
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc  # 记忆编码器的Sigmoid偏置
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc  # 对记忆编码器的点掩码进行二值化
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc  # 记忆编码器中是否使用不重叠的掩码
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval  # 评估时的记忆时间步幅
        # 在有掩码输入的帧上，是否直接输出输入掩码，而不使用SAM提示编码器+掩码解码器
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam  # 不使用SAM时，直接输出掩码
        self.multimask_output_in_sam = multimask_output_in_sam  # SAM中的多掩码输出
        self.multimask_min_pt_num = multimask_min_pt_num  # 多掩码的最小点数
        self.multimask_max_pt_num = multimask_max_pt_num  # 多掩码的最大点数
        self.multimask_output_for_tracking = multimask_output_for_tracking  # 用于跟踪的多掩码输出
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr  # 对象指针使用多掩码token
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid  # IoU预测使用Sigmoid

        # 部分4：SAM风格的提示编码器（用于掩码和点输入）
        # 和SAM风格的掩码解码器，用于最终的掩码输出
        self.image_size = image_size  # 图像大小
        self.backbone_stride = backbone_stride  # 骨干网络步幅
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args  # SAM掩码解码器的额外参数
        self.pred_obj_scores = pred_obj_scores  # 是否预测对象得分
        self.pred_obj_scores_mlp = pred_obj_scores_mlp  # 是否使用MLP预测对象得分
        self.fixed_no_obj_ptr = fixed_no_obj_ptr  # 是否使用固定的无对象指针
        self.soft_no_obj_ptr = soft_no_obj_ptr  # 是否使用软无对象指针
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores  # 如果使用固定无对象指针，必须预测对象得分
            assert self.use_obj_ptrs_in_encoder  # 并且必须在编码器中使用对象指针
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))  # 无对象指针
            trunc_normal_(self.no_obj_ptr, std=0.02)  # 截断正态初始化
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj  # 使用MLP投影对象指针
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()  # 构建SAM头
        self.max_cond_frames_in_attn = max_cond_frames_in_attn  # 注意力中最大条件帧数

        # 模型编译 Model compilation
        if compile_image_encoder:
            # 编译前向函数（而不是整个模块）以允许加载检查点。
            print("启用图像编码器编译。第一次前向传递将会较慢。")
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",  # 最大自动调优模式
                fullgraph=True,  # 全图模式
                dynamic=False,  # 禁用动态模式
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        # forward 方法在此处未实现，并抛出 NotImplementedError 异常，
        # 提示用户应使用 SAM2VideoPredictor 类中的相应方法进行推理，
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference."
            "See notebooks/video_predictor_example.ipynb for an example."
        )

    def _build_sam_heads(self):
        """构建 SAM 风格的提示编码器和掩码解码器"""
        self.sam_prompt_embed_dim = self.hidden_dim  # 设置隐藏维度为提示编码器的嵌入维度
        self.sam_image_embedding_size = self.image_size // self.backbone_stride  # 计算图像嵌入的尺寸，等于输入图像大小除以骨干网络的步幅。

        # 从 SAM 构建 PromptEncoder 和 MaskDecoder
        # （它们的超参数如 `mask_in_chans=16` 来自 SAM 代码）
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,  # 嵌入维度
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),  # 图像嵌入的大小（宽和高）
            input_image_size=(self.image_size, self.image_size),  # 输入图像的大小
            mask_in_chans=16,  # 掩码通道数
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,  # 多掩码输出的数量
            transformer=TwoWayTransformer(
                depth=2,  # Transformer 层数
                embedding_dim=self.sam_prompt_embed_dim,  # Transformer 的嵌入维度
                mlp_dim=2048,  # MLP 的维度
                num_heads=8,  # 多头注意力机制的头数
            ),
            transformer_dim=self.sam_prompt_embed_dim,  # Transformer 的维度
            iou_head_depth=3,  # IoU 预测头的深度
            iou_head_hidden_dim=256,  # IoU 预测头的隐藏层维度
            use_high_res_features=self.use_high_res_features_in_sam,  # 是否使用高分辨率特征
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,  # IoU 预测是否使用 Sigmoid 函数
            pred_obj_scores=self.pred_obj_scores,  # 是否预测对象分数
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,  # 对象分数的 MLP 设置
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,  # 是否使用多掩码 token 进行对象指针预测
            **(self.sam_mask_decoder_extra_args or {}),  # 传递额外的掩码解码器参数
        )
        if self.use_obj_ptrs_in_encoder:
            # 对 SAM 输出 token 进行线性投影，以将其转换为对象指针
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )  # 使用 MLP 进行对象指针投影
        else:
            self.obj_ptr_proj = torch.nn.Identity()  # 否则使用恒等投影
        if self.proj_tpos_enc_in_obj_ptrs:
            # 对时间位置编码进行线性投影，以避免与空间位置编码产生干扰
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()  # 否则使用恒等投影

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM 提示编码器和掩码头的前向传播。

        输入:
        - backbone_features: 形状为 [B, C, H, W] 的图像特征
        - point_inputs: 包含 "point_coords" 和 "point_labels" 的字典，其中：
          1) "point_coords" 形状为 [B, P, 2]，数据类型为 float32，包含 P 个输入点的绝对像素坐标，格式为 (x, y)
          2) "point_labels" 形状为 [B, P]，数据类型为 int32，1 表示正点击，0 表示负点击，-1 表示填充
        - mask_inputs: 形状为 [B, 1, H*16, W*16] 的掩码，类型为 float 或 bool，与图像具有相同的空间大小
        - high_res_features: 可以为 1) None 或 2) 长度为 2 的列表，包含形状为 [B, C, 4*H, 4*W] 和 [B, C, 2*H, 2*W] 的两个特征图，
          将作为 SAM 解码器的高分辨率特征图使用
        - multimask_output: 如果为 True，输出 3 个候选掩码及其对应的 3 个 IoU 估计；如果为 False，则仅输出 1 个掩码及其对应的 IoU 估计

        输出:
        - low_res_multimasks: 形状为 [B, M, H*4, W*4] 的低分辨率掩码输出（M = 3 如果 `multimask_output=True`，M = 1 如果 `multimask_output=False`），
          SAM 输出的低分辨率掩码 logits（在 sigmoid 之前），分辨率为输入骨干特征图的 4 倍（1/4 步幅）
        - high_res_multimasks: 形状为 [B, M, H*16, W*16] 的高分辨率掩码输出（M = 3 如果 `multimask_output=True`，M = 1 如果 `multimask_output=False`），
          从低分辨率掩码上采样得到，与图像大小相同（步幅为 1 像素）
        - ious: 形状为 [B, M] 的 IoU 估计（M = 3 如果 `multimask_output=True`，M = 1 如果 `multimask_output=False`），每个输出掩码的 IoU 估计
        - low_res_masks: 形状为 [B, 1, H*4, W*4] 的最佳低分辨率掩码。
          如果 `multimask_output=True`，则为 IoU 估计最高的掩码。如果 `multimask_output=False`，则与 `low_res_multimasks` 相同。
        - high_res_masks: 形状为 [B, 1, H*16, W*16] 的最佳高分辨率掩码。
          如果 `multimask_output=True`，则为 IoU 估计最高的掩码。如果 `multimask_output=False`，则与 `high_res_multimasks` 相同。
        - obj_ptr: 形状为 [B, C] 的对象指针向量，基于 SAM 掩码解码器的输出 token 提取。
        """
        B = backbone_features.size(0)  # 获取批量大小
        device = backbone_features.device  # 获取设备
        assert backbone_features.size(1) == self.sam_prompt_embed_dim  # 验证特征维度是否匹配嵌入维度
        assert backbone_features.size(2) == self.sam_image_embedding_size  # 验证特征图高度是否匹配图像嵌入大小
        assert backbone_features.size(3) == self.sam_image_embedding_size  # 验证特征图宽度是否匹配图像嵌入大小

        # a) 处理点提示
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]  # 获取点坐标
            sam_point_labels = point_inputs["point_labels"]  # 获取点标签
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B  # 验证点输入的批量大小是否匹配
        else:
            # 如果没有提供点提示，则填充一个空点（标签为 -1）
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) 处理掩码提示
        if mask_inputs is not None:
            # 如果提供了掩码提示，则在需要时将其缩小到低分辨率掩码输入，并作为密集掩码提示输入到 SAM 掩码编码器
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)  # 验证掩码输入形状是否正确
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,  # 调整掩码大小以匹配编码器输入
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # 使用抗锯齿进行下采样
                )
            else:
                sam_mask_prompt = mask_inputs  # 如果大小已经匹配，直接使用输入掩码
        else:
            # 如果没有提供掩码，则传递 None（SAM 的提示编码器将在此情况下添加一个学习的 `no_mask_embed` 以指示没有掩码输入）。
            sam_mask_prompt = None

        # 生成稀疏和密集提示嵌入
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),  # 传递点坐标和标签
            boxes=None,  # 此处未使用 box 提示
            masks=sam_mask_prompt,  # 传递掩码提示
        )
        # 使用 SAM 掩码解码器进行解码
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,  # 传递骨干特征图作为图像嵌入
            image_pe=self.sam_prompt_encoder.get_dense_pe(),  # 获取密集位置编码
            sparse_prompt_embeddings=sparse_embeddings,  # 稀疏提示嵌入
            dense_prompt_embeddings=dense_embeddings,  # 密集提示嵌入
            multimask_output=multimask_output,  # 是否输出多掩码
            repeat_image=False,  # 图像已经批处理，不重复 the image is already batched
            high_res_features=high_res_features,  # 高分辨率特征
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0  # 预测对象是否出现

            # 用于空间记忆的掩码始终是对象和无对象之间的*硬*选择，与实际掩码预测一致
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,  # 如果对象出现，保持原掩码
                NO_OBJ_SCORE,  # 否则使用无对象分数
            )

        # 将掩码从可能的 bfloat16（或 float16）转换为 float32
        # （旧版 PyTorch 在 2.1 之前不支持对 bf16 进行 `interpolate`）
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),  # 上采样到图像大小
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]  # 提取 SAM 输出 token
        if multimask_output:
            # 选择最佳掩码预测（IoU 估计最高）
            best_iou_inds = torch.argmax(ious, dim=-1)   # 获取最佳 IoU 索引
            batch_inds = torch.arange(B, device=device)  # 获取批次索引
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)  # 提取最佳低分辨率掩码
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)  # 提取最佳高分辨率掩码
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # 从 SAM 输出 token 中提取对象指针（处理遮挡情况）
        obj_ptr = self.obj_ptr_proj(sam_output_token)  # 使用线性投影或 MLP 投影来处理 SAM 输出 token
        if self.pred_obj_scores:
            # 允许*软*的无对象指针，与掩码不同 Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                # 只有使用 gt 时才可能进行硬判断
                assert not self.teacher_force_obj_scores_for_mem
                lambda_is_obj_appearing = object_score_logits.sigmoid()  # 使用 sigmoid 计算对象出现的概率
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()  # 将对象出现的硬选择转换为 float 类型

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr  # 计算有对象的对象指针
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr  # 计算无对象的对象指针

        return (
            low_res_multimasks,  # 低分辨率掩码 logits
            high_res_multimasks,  # 高分辨率掩码 logits
            ious,  # 每个掩码的 IoU 估计
            low_res_masks,  # 选择的最佳低分辨率掩码
            high_res_masks,  # 选择的最佳高分辨率掩码
            obj_ptr,  # 对象指针
            object_score_logits,  # 对象分数 logits
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        直接将二进制 `mask_inputs` 转换为输出掩码 logits，而不使用 SAM。
        （输入和输出形状与 _forward_sam_heads 方法相同）。
        """
        # 使用 -10/+10 作为负/正像素的 logits（在 sigmoid 后非常接近 0/1 的概率）。
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()  # 将掩码输入转换为浮点数
        high_res_masks = mask_inputs_float * out_scale + out_bias  # 将掩码输入转换为高分辨率掩码 logits
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # 使用抗锯齿进行下采样
        )
        # 在掩码输入下提供一个全为 1 的虚拟 IoU 预测
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # 作为虚拟对象指针（形状为 [B, C]）的全零张量
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # 使用 SAM 解码器从掩码输入生成对象指针
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # 在此方法中，我们将 mask_input 视为输出，例如直接用于创建空间记忆；
        # 下面，我们遵循相同的设计公理，使用 mask_input 来决定是否出现对象，而不是依赖于 SAM 解码器的 object_scores。
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)  # 判断是否有任何像素大于 0
        is_obj_appearing = is_obj_appearing[..., None]  # 扩展维度以匹配目标形状
        lambda_is_obj_appearing = is_obj_appearing.float()  # 将是否出现对象转换为浮点数
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias  # 计算对象分数 logits
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr  # 计算有对象的对象指针
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr  # 计算无对象的对象指针

        return (
            low_res_masks,  # 低分辨率掩码 logits
            high_res_masks,  # 高分辨率掩码 logits
            ious,  # 每个掩码的 IoU 估计
            low_res_masks,  # 选择的最佳低分辨率掩码
            high_res_masks,  # 选择的最佳高分辨率掩码
            obj_ptr,  # 对象指针
            object_score_logits,  # 对象分数 logits
        )

    def forward_image(self, img_batch: torch.Tensor):
        """获取输入批次的图像特征。"""
        backbone_out = self.image_encoder(img_batch)  # 使用图像编码器获取图像特征
        if self.use_high_res_features_in_sam:
            # 预先计算 SAM 解码器中的级别 0 和级别 1 特征，以避免在每次 SAM 点击时再次运行
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out  # 返回处理后的特征

    def _prepare_backbone_features(self, backbone_out):
        """准备并扁平化视觉特征。"""
        backbone_out = backbone_out.copy()  # 复制特征字典以避免修改原始数据
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]  # 取最后 `num_feature_levels` 层的特征图
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]  # 取最后 `num_feature_levels` 层的位置编码

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]  # 获取每层特征图的尺寸
        # 将 NxCxHxW 展平为 HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,  # 是否是初始化条件帧
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # 逆向跟踪时间顺序（用于演示）
        preloading_memory_cond_frame_idx=None,  # 预加载的记忆条件帧索引
    ):
        """将当前帧的视觉特征图与之前的记忆融合。"""
        # print(f"SAM2Base类_prepare_memory_conditioned_features"
        #       f"is_init_cond_frame:{is_init_cond_frame}")  # False

        # print(f"frame_idx:{frame_idx},is_init_cond_frame:{is_init_cond_frame}")

        B = current_vision_feats[-1].size(1)  # 当前帧的批次大小
        C = self.hidden_dim  # 隐藏维度
        H, W = feat_sizes[-1]  # 顶层（最低分辨率）特征大小
        device = current_vision_feats[-1].device  # 当前设备
        # `self.num_maskmem == 0` 的情况主要用于在图像上重现 SAM。
        # 在这种情况下，我们跳过与任何记忆的融合。
        if self.num_maskmem == 0:  # 禁用记忆并跳过融合
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # 步骤 1：将当前帧的视觉特征条件化于先前的记忆
        if not is_init_cond_frame:
            # print(f"结合当前帧之前的记忆帧和对象指针信息，为后续帧的处理构建一个完整的记忆特征与位置编码列表")
            # 检索用 maskmem 编码的记忆
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # 首先添加条件帧的输出（所有条件帧的 t_pos=0）
            assert len(output_dict["cond_frame_outputs"]) > 0
            # 选择时间上最接近的条件帧用于交叉注意
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn, preloading_memory_cond_frame_idx
            )

            # # 打印 selected_cond_outputs 中的每个张量的形状
            # for i, out in enumerate(selected_cond_outputs.values()):
            #     print(f"i:{i},out:{out['maskmem_features'].shape}")

            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]

            # 在当前帧之前添加最后 (self.num_maskmem - 1) 帧作为非条件记忆
            # 最早的一帧 t_pos=1，最新的一帧 t_pos=self.num_maskmem-1
            # 还允许非连续地选择记忆帧（r>1），在这种情况下，选择每 r 帧中的 (self.num_maskmem - 2) 帧以及最后一帧
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # 当前帧之前多少帧
                if t_rel == 1:
                    # 对于 t_rel == 1，取上一帧（无论 r 如何）
                    if not track_in_reverse:
                        # 当前帧之前的一帧（即 frame_idx - 1）
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # 当前帧之后的一帧（即 frame_idx + 1）
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # 对于 t_rel >= 2，从每 r 帧中选择记忆帧
                    if not track_in_reverse:
                        # 首先找到当前帧之前每 r 帧中的最近一帧
                        # 对于 r=1，这将是 (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # 然后在每 r 帧中进一步查找
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # 首先找到当前帧之后每 r 帧中的最近一帧
                        # 对于 r=1，这将是 (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # 然后在每 r 帧中进一步查找
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                # print(f"获取非条件帧t_pos：{t_pos}，prev_frame_idx：{prev_frame_idx}")
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # print("out is None")
                    # 如果未选择的条件帧在最后 (self.num_maskmem - 1) 帧中，我们仍然将其视为非条件帧。
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # 跳过填充帧

                # 打印 prev 中的每个张量的形状
                # print(f"t_pos: {t_pos}")
                # print(f"maskmem_features shape: {prev['maskmem_features'].shape}")
                # print(f"maskmem_pos_enc shape: {prev['maskmem_pos_enc'][-1].shape}")

                # "maskmem_features" 可能已经在演示使用案例中转移到 CPU，
                # 因此我们将其加载回 GPU（如果已经在 GPU 上，则不操作）。
                feats = prev["maskmem_features"].to(device, non_blocking=True)

                feats = feats.to(torch.float32)  # 确保是 float32, 混合精度推理下这里是自动转换
                # print(f"feats.dtype:{feats.dtype}")  # torch.bfloat16会导致后续报错
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # 空间位置编码（可能已经在评估时转移到 CPU）
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # 时间位置编码
                maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1])
                to_cat_memory_pos_embed.append(maskmem_enc)

            # 构建过去对象指针的列表
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # 首先添加来自选定条件帧的对象指针 （在评估期间可选，仅包括过去的对象指针）
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # 时间位置编码包含每个指针距离当前帧的距离
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # 添加到 (max_obj_ptrs_in_encoder - 1) 的非条件帧
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # # 如果有至少一个对象指针，将其添加到交叉注意中
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # 将对象指针沿 dim=0 堆叠到 [ptr_seq_len, B, C] 形状
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # 基于每个对象指针距离当前帧的时间位置嵌入（以正弦嵌入进行归一化，归一化因子为最大指针数）
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # 将指针拆分为 (C // self.mem_dim) 个 token，用于 self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    # print(f"obj_ptrs:{obj_ptrs.shape},obj_pos:{obj_pos.shape}")
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # 对于初始条件帧，不使用任何先前的记忆进行编码
            if self.directly_add_no_mem_embed:
                # print(f"初始条件帧，不使用任何先前的记忆进行编码")
                # 直接添加无记忆嵌入（而不是使用变换器编码器）
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

                return pix_feat_with_mem

            # 在第一帧使用一个虚拟标记（以避免将空内存输入到 Transformer 编码器中）
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # 第2步：将记忆拼接起来，并通过 Transformer 编码器进行前向传播
        # for idx, tensor in enumerate(to_cat_memory):
        #     print(f"Tensor {idx} shape: {tensor.shape}")
        # for idx, tensor in enumerate(to_cat_memory_pos_embed):
        #     print(f"to_cat_memory_pos_embed[{idx}] shape: {tensor.shape}")
        memory = torch.cat(to_cat_memory, dim=0)  # 沿着第0维（时间维度）拼接记忆
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)  # 沿着第0维拼接记忆位置嵌入

        # print(f"to_cat_memory:{len(to_cat_memory)}")
        # print(f"memory.shape:{memory.shape}")
        # print(f"current_vision_feats.shape:{current_vision_feats[-1].shape}")  # 恒定为torch.Size([4096, 1, 256])
        # memory和memory_pos_embed形状一致torch.Size([N, Batch, 64]) N为条件帧数量*4090

        # 使用 memory_attention 层处理当前视觉特征和记忆
        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,  # 当前帧的视觉特征
            curr_pos=current_vision_pos_embeds,  # 当前帧的视觉位置嵌入
            memory=memory,  # 记忆特征
            memory_pos=memory_pos_embed,  # 记忆位置嵌入
            num_obj_ptr_tokens=num_obj_ptr_tokens,  # 对象指针标记数量
        )
        # 释放memory_attention产生的不再使用的显存,为了防止预加载内存库时异常显存开销！很重要
        torch.cuda.empty_cache()  # TODO：具体清除的是哪些部分?为什么会带来大量临时显存开销？
        # 重新调整输出形状 (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)  # 转换为 [B, C, H, W] 形状
        # pix_feat_with_mem占用约64MB显存资源

        return pix_feat_with_mem  # 返回带有记忆的视觉特征

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """将当前图像及其预测编码为记忆特征。"""
        B = current_vision_feats[-1].size(1)  # 当前帧的批量大小
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # 顶层（最低分辨率）特征大小

        # 顶层特征，(HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # 可选：在评估期间应用不重叠的掩码约束（仅在批量大小为1的情况下使用，所有对象来自同一视频）
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # 在应用 sigmoid 之前，用温度缩放原始掩码 logits
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            # 如果二值化且不是训练阶段，将掩码转换为 0 或 1
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # 否则，应用 sigmoid 将原始掩码 logits 转换到 (0, 1) 范围内
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        # 对 sigmoid 概率应用缩放和偏置项
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        # 使用 memory_encoder 编码视觉特征和掩码
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # 已经应用了 sigmoid
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # 在空间记忆中添加一个无物体嵌入，以表示该帧被预测为被遮挡的（即帧中没有物体出现）
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc


    def _track_step(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            preloading_memory_cond_frame_idx,
    ):
        # print(f"---track_step中track_in_reverse：{track_in_reverse}---")
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        # 高分辨率特征图，用于 SAM 头，重塑 (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # 当 use_mask_input_as_output_without_sam=True 时，直接输出掩码输入
            # （视为 GT 掩码），而不使用 SAM 提示编码器和掩码解码器
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # 将视觉特征与记忆库中的先前记忆特征融合
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                preloading_memory_cond_frame_idx=preloading_memory_cond_frame_idx, # 预加载的记忆条件帧索引
            )
            # 应用 SAM 风格的分割头
            # 在此，我们可能将之前预测的低分辨率 SAM 掩码 logits 输入到 SAM 掩码解码器，
            # 例如在演示中，这些 logits 来自早期交互而不是修正采样
            # （在这种情况下，任何 `mask_inputs` 不应该到达这里，因为它们已被发送到 _use_mask_as_output）
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat


    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        # 最后对预测掩码运行记忆编码器，将其编码为新的记忆特征（可用于未来的帧）
        if run_mem_encoder and self.num_maskmem > 0:
            # 将高分辨率掩码用于记忆编码器
            high_res_masks_for_mem_enc = high_res_masks

            # 调用 _encode_new_memory 方法对当前视觉特征和高分辨率掩码进行编码，生成新的记忆特征
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            # 将生成的记忆特征和位置编码添加到输出字典中
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def print_gpu_memory(self):  # TODO:暂时用于开发时查看显存使用情况，后续删除
        try:
            # 使用 nvidia-smi 获取显存信息
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"])
            result = result.decode("utf-8").strip().split("\n")
            # 每个 GPU 的显存使用情况 (used, free)
            gpu_memory = [tuple(map(int, line.split(", "))) for line in result]
            if gpu_memory:
                for idx, (used, free) in enumerate(gpu_memory):
                    print(f"GPU{idx}显存 - 使用: {used} MB, 空余: {free} MB")
        except Exception as e:
            print(f"Error in getting GPU memory: {e}")
            return None

    def track_step(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse=False,
            run_mem_encoder=True,
            # 之前预测的 SAM 掩码 logits（可以与新点击一起输入到演示中）。
            prev_sam_mask_logits=None,
            preloading_memory_cond_frame_idx=None,  # 预加载内存库中的条件帧索引
    ):

        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            preloading_memory_cond_frame_idx,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # 仅在推理时添加对象分数 logits（避免在激活检查点中引入未使用的参数；
            # 主要用于演示中，以使用整合的掩码编码空间记忆）
            current_out["object_score_logits"] = object_score_logits

        # 最后运行记忆编码器，将预测的掩码编码为新的记忆特征（可用于未来帧）
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out


    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """是否在SAM头中使用多掩码输出。"""
        # 获取点输入的数量，如果点输入为空，则数量为0
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        # 判断是否需要使用多掩码输出的条件
        multimask_output = (
            self.multimask_output_in_sam  # 是否启用多掩码输出
            and (is_init_cond_frame or self.multimask_output_for_tracking)  # 是否为初始条件帧或跟踪模式
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)  # 点数量在指定范围内
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        对 `pred_masks` 中的对象分数应用非重叠约束。在这里，我们保留 `pred_masks` 中每个空间位置的最高评分对象。
        """
        batch_size = pred_masks.size(0)  # 获取批次大小
        if batch_size == 1:
            # 如果批次大小为1，直接返回 `pred_masks`
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": 在每个位置上具有最高分数的对象索引
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": `pred_masks` 中每个对象切片的对象索引（沿着维度0）
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # 保留最高分数的对象索引，其他索引则被压制
        # 将重叠区域的分数压制到 -10.0 以避免重叠区域的分数（在这里，sigmoid(-10.0)=4.5398e-05）
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
