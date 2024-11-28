# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,  # Transformer的通道维度
        transformer: nn.Module,  # 用于掩码预测的transformer模块
        num_multimask_outputs: int = 3,  # 用于多掩码输出的掩码数量
        activation: Type[nn.Module] = nn.GELU,  # 上采样掩码时使用的激活函数
        iou_head_depth: int = 3,  # 用于预测掩码质量的MLP的深度
        iou_head_hidden_dim: int = 256,  # 用于预测掩码质量的MLP的隐藏维度
        use_high_res_features: bool = False,  # 是否使用高分辨率特征
        iou_prediction_use_sigmoid=False,  # 是否在IoU预测中使用Sigmoid
        dynamic_multimask_via_stability=False,  # 是否通过稳定性动态选择多掩码
        dynamic_multimask_stability_delta=0.05,  # 动态多掩码稳定性的变化值
        dynamic_multimask_stability_thresh=0.98,  # 动态多掩码稳定性的阈值
        pred_obj_scores: bool = False,  # 是否预测对象得分
        pred_obj_scores_mlp: bool = False,  # 对象得分是否使用MLP预测
        use_multimask_token_for_obj_ptr: bool = False,  # 是否使用多掩码Token作为对象指针
    ) -> None:
        """
        根据图像和提示嵌入来预测掩码，使用的是transformer架构。

        参数:
          transformer_dim (int): transformer的通道维度
          transformer (nn.Module): 用于预测掩码的transformer
          num_multimask_outputs (int): 在去歧义掩码时要预测的掩码数量
          activation (nn.Module): 用于上采样掩码时的激活函数类型
          iou_head_depth (int): 用于预测掩码质量的MLP的深度
          iou_head_hidden_dim (int): 用于预测掩码质量的MLP的隐藏维度
        """
        super().__init__()  # 调用父类的初始化方法
        self.transformer_dim = transformer_dim  # 保存transformer的通道维度
        self.transformer = transformer  # 保存transformer模块

        self.num_multimask_outputs = num_multimask_outputs  # 保存多掩码输出数量

        self.iou_token = nn.Embedding(1, transformer_dim)  # 定义IoU嵌入层
        self.num_mask_tokens = num_multimask_outputs + 1  # 定义掩码Token数量
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)  # 定义掩码Token嵌入层

        self.pred_obj_scores = pred_obj_scores  # 是否预测对象得分的标志
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)  # 定义对象得分嵌入层
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr  # 是否使用多掩码Token作为对象指针的标志

        self.output_upscaling = nn.Sequential(  # 定义掩码上采样模块
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2  # 反卷积层，将通道数减小4倍，大小增加2倍
            ),
            LayerNorm2d(transformer_dim // 4),  # 对通道进行LayerNorm
            activation(),  # 使用激活函数
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2  # 再次反卷积，将通道数减小2倍
            ),
            activation(),  # 再次使用激活函数
        )
        self.use_high_res_features = use_high_res_features  # 是否使用高分辨率特征的标志
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1  # 定义一个1x1卷积，将通道数减小8倍
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1  # 定义另一个1x1卷积，将通道数减小4倍
            )

        self.output_hypernetworks_mlps = nn.ModuleList(  # 定义用于掩码输出的MLP列表
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)  # 每个MLP将transformer_dim映射到较小的维度
                for i in range(self.num_mask_tokens)  # 生成与掩码Token数量相同的MLP
            ]
        )

        self.iou_prediction_head = MLP(  # 定义用于IoU预测的MLP
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,  # 是否在输出时使用Sigmoid
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)  # 定义用于预测对象得分的线性层
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)  # 如果需要，使用MLP代替线性层

        # 当输出单个掩码时，如果单掩码的稳定性得分低于阈值，可以动态地使用最佳多掩码输出
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability  # 是否动态选择多掩码的标志
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta  # 多掩码稳定性的变化值
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh  # 多掩码稳定性的阈值


    def forward(
        self,
        image_embeddings: torch.Tensor,  # 输入的图像嵌入
        image_pe: torch.Tensor,  # 图像的位置信息编码
        sparse_prompt_embeddings: torch.Tensor,  # 稀疏提示的嵌入
        dense_prompt_embeddings: torch.Tensor,  # 稠密提示的嵌入
        multimask_output: bool,  # 是否输出多个掩码
        repeat_image: bool,  # 是否重复图像数据
        high_res_features: Optional[List[torch.Tensor]] = None,  # 高分辨率特征的可选列表
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回值是掩码和IoU预测的元组
        """
        根据图像和提示嵌入来预测掩码。

        参数:
          image_embeddings (torch.Tensor): 图像编码器的输出嵌入
          image_pe (torch.Tensor): 与图像嵌入形状相同的位置信息编码
          sparse_prompt_embeddings (torch.Tensor): 点和框的嵌入
          dense_prompt_embeddings (torch.Tensor): 掩码输入的嵌入
          multimask_output (bool): 是否返回多个掩码或单个掩码

        返回:
          torch.Tensor: 预测的掩码
          torch.Tensor: 预测的掩码质量（IoU）
          torch.Tensor: 用于掩码输出的SAM token
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(  # 调用predict_masks方法，获取掩码、IoU预测、掩码Token和对象得分
            image_embeddings=image_embeddings,  # 传入图像嵌入
            image_pe=image_pe,  # 传入图像位置信息编码
            sparse_prompt_embeddings=sparse_prompt_embeddings,  # 传入稀疏提示嵌入
            dense_prompt_embeddings=dense_prompt_embeddings,  # 传入稠密提示嵌入
            repeat_image=repeat_image,  # 是否重复图像数据
            high_res_features=high_res_features,  # 传入高分辨率特征
        )

        # 根据multimask_output参数选择正确的掩码输出
        if multimask_output:
            masks = masks[:, 1:, :, :]  # 如果输出多个掩码，则选择第2个及以后的掩码
            iou_pred = iou_pred[:, 1:]  # 对应的IoU预测也选择第2个及以后的结果
        elif self.dynamic_multimask_via_stability and not self.training:
            # 在非训练模式下，根据稳定性动态选择掩码
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]  # 否则只选择第一个掩码
            iou_pred = iou_pred[:, 0:1]  # 对应的IoU预测也只选择第一个结果

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            # 如果需要多掩码输出且使用多掩码Token，选择第2个及以后的掩码Token
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # 选择单个掩码输出Token
            # 测试时，即使我们跟踪1次点击并使用multimask_output=True，
            # 我们仍然在这里使用单掩码Token。原因是训练时我们总是跟踪多个点击，
            # 所以训练时看到的过去的Tokens总是单掩码Token（并且我们让它成为对象内存Token）。
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测掩码。有关更多详细信息，请参见 'forward' 方法。"""
        # 拼接 Output Token
        s = 0
        if self.pred_obj_scores:
            # 如果预测对象得分，将对象得分Token、IoU Token和掩码Token拼接在一起
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            # 否则，只拼接IoU Token和掩码Token
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        # 扩展Token以匹配批量大小
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 将每图像数据扩展为每掩码
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings

        assert (image_pe.size(0) == 1), "image_pe 应该在 batch 维度上有尺寸 1 (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]  # IoU Token 输出
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # 掩码Token 输出

        # 上采样掩码嵌入并使用掩码Token预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            # 如果不使用高分辨率特征，则使用输出上采样层
            upscaled_embedding = self.output_upscaling(src)
        else:
            # 使用高分辨率特征进行上采样
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # 为每个掩码Token生成超网络输入
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        # 使用超网络输入和上采样嵌入预测掩码
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            # 如果预测对象得分，则生成对象得分Logits
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # 对象得分Logits - 默认为10.0，即假设对象存在，sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        计算掩码Logits的稳定性得分，基于上限和下限阈值之间的IoU，
        类似于 https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = mask_logits.flatten(-2)  # 将mask_logits在倒数第二维度上展平，以便在计算稳定性得分时按掩码计算
        stability_delta = self.dynamic_multimask_stability_delta  # 获取用于计算稳定性得分的稳定性阈值
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()  # 计算掩码区域大于稳定性阈值的面积
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()  # 计算掩码区域大于负稳定性阈值的面积
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)  # 计算稳定性得分，如果area_u为0则稳定性得分为1.0
        return stability_scores  # 返回稳定性得分

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        当输出单个掩码时，如果当前单掩码输出的稳定性得分（基于输出Token 0）低于阈值，
        我们将从多掩码输出（基于输出Token 1~3）中选择预测IoU得分最高的掩码。
        这旨在确保在点击和跟踪时都有有效的掩码。
        """
        # 从多掩码输出Token（1~3）中选择最佳掩码
        multimask_logits = all_mask_logits[:, 1:, :, :]  # 提取所有多掩码的logits，忽略第一个掩码Token（0）
        multimask_iou_scores = all_iou_scores[:, 1:]  # 提取所有多掩码的IoU得分，忽略第一个掩码Token（0）
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)  # 找到每个样本中IoU得分最高的掩码的索引
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )  # 创建批次索引，用于从多掩码logits中选择最佳掩码
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]  # 根据最佳索引提取最佳的多掩码logits
        best_multimask_logits = best_multimask_logits.unsqueeze(1)  # 增加一个维度以匹配单掩码输出的形状
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]  # 根据最佳索引提取最佳的多掩码IoU得分
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)  # 增加一个维度以匹配单掩码输出的形状

        # 单掩码输出Token 0及其稳定性得分
        singlemask_logits = all_mask_logits[:, 0:1, :, :]  # 提取单掩码的logits（Token 0）
        singlemask_iou_scores = all_iou_scores[:, 0:1]  # 提取单掩码的IoU得分（Token 0）
        stability_scores = self._get_stability_scores(singlemask_logits)  # 计算单掩码的稳定性得分
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh  # 判断单掩码的稳定性是否达到阈值

        # 动态回退到最佳的多掩码输出，以应对低稳定性得分
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),  # 如果单掩码稳定，则保持单掩码logits，否则使用最佳的多掩码logits
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),  # 如果单掩码稳定，则保持单掩码IoU得分，否则使用最佳的多掩码IoU得分
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out  # 返回最终的掩码logits和IoU得分
