
import os
import json
from tqdm import tqdm
from itertools import product

from det_sam2_RT import VideoProcessor
from postprocess_det_sam2 import VideoPostProcessor

class EvalDetSAM2PostProcess:
    def __init__(
        self,
        # Det-SAM2推理主干参数
        sam2_output_frame_dir,  # 用来存储sam2的输出掩码的帧的文件夹
        sam2_checkpoint_path,  # SAM2的模型权重
        sam2_config_path,  # SAM2的配置文件
        detect_model_weights,  # 检测模型权重
        load_inference_state_path=None,  # 预加载内存库
        # 临时的中间结果
        temp_video_segments_pkl = "/root/autodl-tmp/temp_output/video_segments.pkl",
        temp_special_classes_detection_pkl = "/root/autodl-tmp/temp_output/special_classes_detection.pkl",
        # 后处理可视化路径
        visualize_result_dir=None,
    ):
        self.sam2_output_frame_dir = sam2_output_frame_dir
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_config_path = sam2_config_path
        self.detect_model_weights = detect_model_weights
        self.load_inference_state_path = load_inference_state_path

        # 初始化Det-SAM2实时推理主干的VideoProcessor类
        self.video_processor = VideoProcessor(
            output_dir=self.sam2_output_frame_dir,
            sam2_checkpoint=self.sam2_checkpoint_path,
            model_cfg=self.sam2_config_path,
            detect_model_weights=self.detect_model_weights,
            load_inference_state_path=self.load_inference_state_path,
        )

        # 初始化Det-SAM2的后处理类
        self.post_processor = VideoPostProcessor()
        # 临时中间结果存放路径
        self.temp_video_segments_pkl = temp_video_segments_pkl
        self.temp_special_classes_detection_pkl = temp_special_classes_detection_pkl
        # 后处理可视化路径
        self.visualize_result_dir = visualize_result_dir
        if self.visualize_result_dir is not None:
            os.makedirs(self.visualize_result_dir, exist_ok=True)

    def reset_Processor(
            self,
            detect_confidence,  # 检测模型置信度阈值
            frame_buffer_size,  # 累积帧数
            detect_interval,  # 检测间隔
            max_frame_num_to_track,  # 视频传播最大长度
            max_inference_state_frames,  # 内存库最大帧数
            load_inference_state_path,  # 预加载内存库路径
            pot_distance_threshold,  # 袋口附近距离阈值
            pot_velocity_threshold,  # 袋口附近速度阈值
            ball_distance_threshold,  # 球间撞击的距离判断阈值
            ball_velocity_threshold,  # 球间撞击的速度变化判断阈值
            table_margin,  # 桌子边界的缓冲值
            rebound_velocity_threshold,  # 判断反弹的速度阈值
    ):
        '''
        为VideoProcessor和PostProcessor加载新的参数组合
        '''
        # 释放旧的 VideoProcessor 实例
        del self.video_processor
        # 以新的参数初始化Det-SAM2实时推理主干的VideoProcessor类
        self.video_processor = VideoProcessor(
            output_dir=self.sam2_output_frame_dir,
            sam2_checkpoint=self.sam2_checkpoint_path,
            model_cfg=self.sam2_config_path,
            detect_model_weights=self.detect_model_weights,
            load_inference_state_path=load_inference_state_path,
            detect_confidence=detect_confidence,  # 检测模型置信度阈值
            frame_buffer_size=frame_buffer_size, # 每次累积多少帧后执行推理
            detect_interval=detect_interval, # 每隔多少帧执行一次检测（条件帧提示）, -1时不进行检测（不添加条件帧提示）但是需要所有帧中必须至少存在一帧条件帧（可以是系统提示帧）
            max_frame_num_to_track=max_frame_num_to_track,  # 每次传播（propagate_in_video）时从最后一帧开始执行倒序追踪推理的最大长度。该限制可以有效降低长视频的重复计算开销
            max_inference_state_frames=max_inference_state_frames,  # 不应小于max_frame_num_to_track ; 限制inference_state中的帧数超出这个帧数时会不断释放掉旧的帧，-1表示不限制
        )
        # 释放旧的 PostProcessor 实例
        del self.post_processor
        # 以新的参数初始化Det-SAM2的后处理类
        self.post_processor = VideoPostProcessor(
            pot_distance_threshold=pot_distance_threshold,
            pot_velocity_threshold=pot_velocity_threshold,
            ball_distance_threshold=ball_distance_threshold,
            ball_velocity_threshold=ball_velocity_threshold,
            table_margin=table_margin,
            rebound_velocity_threshold=rebound_velocity_threshold,
        )

    def eval_all_settings(
        self,
        videos_dir,  # 评估集视频文件夹
        eval_jsonl_path,  # 评估集标注文本jsonl文件路径
        eval_output_dir,  # 评估集结果输出文件夹

        detect_confidence_list,  # 检测模型置信度阈值参数列表
        frame_buffer_size_list,  # 累积帧数参数列表
        detect_interval_list,  # 检测间隔参数列表
        max_frame_num_to_track_list,  # 视频传播最大长度参数列表
        max_inference_state_frames_list,  # 限制inference_state中保留最大帧数参数列表
        load_inference_state_path_list,  # 预加载内存库路径参数列表
        pot_distance_threshold_list,  # 袋口附近距离阈值参数列表
        pot_velocity_threshold_list,  # 袋口附近速度阈值参数列表
        ball_distance_threshold_list,  # 球间撞击的距离判断阈值参数列表
        ball_velocity_threshold_list,  # 球间撞击的速度变化判断阈值参数列表
        table_margin_list,  # 桌子边界的缓冲值参数列表
        rebound_velocity_threshold_list,  # 判断反弹的速度阈值参数列表
    ):
        """
        尝试所有参数组合下的评估
        """
        # 定义所有参数列表
        param_grid = {
            "frame_buffer_size": frame_buffer_size_list,
            "detect_interval": detect_interval_list,
            "max_frame_num_to_track": max_frame_num_to_track_list,
            "max_inference_state_frames": max_inference_state_frames_list,
            "load_inference_state_path": load_inference_state_path_list,
            "pot_distance_threshold": pot_distance_threshold_list,
            "pot_velocity_threshold": pot_velocity_threshold_list,
            "ball_distance_threshold": ball_distance_threshold_list,
            "ball_velocity_threshold": ball_velocity_threshold_list,
            "table_margin": table_margin_list,
            "rebound_velocity_threshold": rebound_velocity_threshold_list,
            "detect_confidence": detect_confidence_list,
        }

        # 使用 itertools.product 生成参数组合
        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))

            # 跳过不符合条件的组合
            if params["max_frame_num_to_track"] < params["frame_buffer_size"]: # 传播最大长度必须大于等于累积帧数,否则会存在帧未被推理
                continue
            if params["detect_interval"] == 0 and params["load_inference_state_path"] is None: # 当检测间隔为0时且不预加载内存库时,将不存在任何条件帧无法进行正常推理
                continue
            if params["max_inference_state_frames"] != -1 and params["max_inference_state_frames"] < params["max_frame_num_to_track"]: # 如果启用最大内存库帧数限制,那么最大帧数限制应当大于最大追踪（视频传播）帧数
                continue
            print("当前正在评估的参数组合：", params)

            # 为VideoProcessor加载新的参数组合
            self.reset_Processor(**params)

            # 以当前参数组合遍历一遍评估集
            avg_eval_results = self.eval(videos_dir, eval_jsonl_path, eval_output_dir)
            print(f"当前参数组合下的评估结果：{avg_eval_results}")

            # 将统计的评估结果追加到 eval_results.json 文件中，记录当前参数组合下的得分
            eval_output_path = os.path.join(eval_output_dir, "eval_results.json")
            os.makedirs(eval_output_dir, exist_ok=True)

            # 如果文件已存在，则先读取内容
            if os.path.exists(eval_output_path):
                with open(eval_output_path, 'r', encoding='utf-8') as infile:
                    eval_results_data = json.load(infile)
            else:
                eval_results_data = []

            # 将新的结果追加到列表
            eval_results_data.append({
                "params_setting": params,
                "average_results": avg_eval_results,
            })

            # 将更新后的内容写回文件
            with open(eval_output_path, 'w', encoding='utf-8') as outfile:
                json.dump(eval_results_data, outfile, indent=4, ensure_ascii=False)

        print(f"已完成所有参数组合的评估,评估结束！")


    def eval(
        self,
        videos_dir,  # 评估集视频文件夹
        eval_jsonl_path,  # 评估集标注文本jsonl文件路径
        eval_output_dir,  # 评估集结果输出文件夹

    ):
        '''
        以当前参数组合遍历一遍评估集
        '''
        # 创建输出文件夹
        os.makedirs(eval_output_dir, exist_ok=True)

        # 初始化用于存储每个视频评估结果的字典
        eval_results = {}

        # 读取并遍历评估集标注
        with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 解析每一行JSONL数据
                    annotation = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"错误出现在{line}行:{e}")

                video_name = annotation.get("video")  # 获取视频名称
                print(f"-正在评估视频: {video_name} ...............................")
                video_path = os.path.join(videos_dir, video_name)

                # 执行Det-SAM2推理主干
                self.video_processor.run(
                    video_path=video_path,
                    output_video_segments_pkl_path=self.temp_video_segments_pkl,
                    output_special_classes_detection_pkl_path=self.temp_special_classes_detection_pkl,
                )

                # 清空Det-SAM2推理主干缓存
                self.video_processor.clear()

                # 执行Det-SAM2后处理
                self.post_processor.get_hole_name(self.temp_special_classes_detection_pkl)  # 将袋口坐标分配給指定袋口
                self.post_processor.get_boundary_from_holes()  # 根据袋口坐标计算桌面有效边界
                self.post_processor.run(self.temp_video_segments_pkl)  # 执行后处理条件判断主干函数
                # 在视频中可视化
                if self.visualize_result_dir is not None:
                    self.post_processor.visualize(video_path, self.visualize_result_dir, video_name)

                # 评估进球检测结果
                pot_gt = annotation.get("pot")  # 评估集GroundTruth
                pot_test = self.post_processor.disappeared_balls  # 模型输出结果
                pot_precision, pot_recall, pot_f1 = self.pot_eval_metrics(pot_gt, pot_test)

                # 评估球间碰撞检测结果
                collision_gt = annotation.get("collision")  # 评估集GroundTruth
                collision_test = self.post_processor.ball_collision  # 模型输出结果
                collision_precision, collision_recall, collision_f1 = self.collision_eval_metrics(collision_gt, collision_test)

                # 评估桌边反弹检测结果
                rebound_gt = annotation.get("rebound")  # 评估集GroundTruth
                rebound_test = self.post_processor.ball_rebound  # 模型输出结果
                rebound_precision, rebound_recall, rebound_f1 = self.rebound_eval_metrics(rebound_gt, rebound_test)

                # 将当前视频的结果存入字典
                eval_results[video_name] = {
                    "pot": {"precision": pot_precision, "recall": pot_recall, "f1": pot_f1},
                    "collision": {"precision": collision_precision, "recall": collision_recall, "f1": collision_f1},
                    "rebound": {"precision": rebound_precision, "recall": rebound_recall, "f1": rebound_f1},
                }

                print(f"-进球检测：查准率：{pot_precision:.4f}，查全率：{pot_recall:.4f}，F1分数：{pot_f1:.4f}")
                print(f"-球间碰撞：查准率：{collision_precision:.4f}，查全率：{collision_recall:.4f}，F1分数：{collision_f1:.4f}")
                print(f"-桌边反弹：查准率：{rebound_precision:.4f}，查全率：{rebound_recall:.4f}，F1分数：{rebound_f1:.4f}")

                # 清空Det-SAM2后处理缓存
                self.post_processor.clear()

        # 统计所有样本中最终平均得分
        avg_eval_results = self.calulate_avg_metrics(eval_results)

        # 将详细的评估结果写入文件, 记保存每个样本的详细得分
        # eval_output_path = os.path.join(eval_output_dir, "eval_results.json")

        # 返回最终平均得分
        return avg_eval_results


    def precision_recall_f1_score(self, true_set, pred_set):
        """计算查准率、查全率和 F1 分数."""
        # 如果真值和预测都是空集合，返回 100% 的查准率和查全率
        if not true_set and not pred_set:
            return 1.0, 1.0, 1.0

        tp = len(true_set & pred_set)  # 真正例：预测正确的个数
        fp = len(pred_set - true_set)  # 假正例：预测错误的个数
        fn = len(true_set - pred_set)  # 假负例：遗漏的正确个数

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return precision, recall, f1

    def pot_eval_metrics(self, pot_gt, pot_test):
        """计算进球检测评估指标"""
        # 转换进球结果为集合形式
        pot_gt_set = set([(int(ball_id), hole) for ball_id, hole in pot_gt.items()])
        pot_test_set = set([(ball_id, data["hole"]) for ball_id, data in pot_test.items()])
        print(f"pot_gt_set:{pot_gt_set}")
        print(f"pot_test_set:{pot_test_set}")

        # 计算查准率、查全率和 F1 分数
        precision_pot, recall_pot, f1_pot = self.precision_recall_f1_score(pot_gt_set, pot_test_set)

        return precision_pot, recall_pot, f1_pot

    def collision_eval_metrics(self, collision_gt, collision_test):
        """计算球间碰撞检测评估指标"""
        # 转换球间碰撞结果为集合形式
        collision_gt_set = set([tuple(sorted(pair)) for pair in collision_gt])
        # 在模型预测输出中将每对球的顺序进行排序，以确保例如(0, 3)和(3, 0)统一为(0, 3)的形式。这样可以避免由于顺序不同而导致的重复错误。
        collision_test_set = set([tuple(sorted(pair)) for pairs in collision_test.values() for pair in pairs])
        print(f"collision_gt_set:{collision_gt_set}")
        print(f"collision_test_set:{collision_test_set}")

        # 计算查准率、查全率和 F1 分数
        precision_collision, recall_collision, f1_collision = self.precision_recall_f1_score(collision_gt_set, collision_test_set)

        return precision_collision, recall_collision, f1_collision

    def rebound_eval_metrics(self, rebound_gt, rebound_test):
        """计算桌边反弹检测评估指标"""
        # 转换桌边反弹结果为集合形式
        rebound_gt_set = set((int(ball_id), side) for ball_id, sides in rebound_gt.items() for side in sides)
        rebound_test_set = set((ball_id, side) for frame_data in rebound_test.values() for ball_id, side in frame_data)
        print(f"rebound_gt_set:{rebound_gt_set}")
        print(f"rebound_test_set:{rebound_test_set}")

        # 计算查准率、查全率和 F1 分数
        precision_rebound, recall_rebound, f1_rebound = self.precision_recall_f1_score(rebound_gt_set, rebound_test_set)

        return precision_rebound, recall_rebound, f1_rebound

    def calulate_avg_metrics(self, eval_results):
        """计算所有视频的平均评估指标"""
        # 计算所有样本中进球评价指标得分
        avg_precision_pot = sum([result["pot"]["precision"] for result in eval_results.values()]) / len(eval_results)
        avg_recall_pot = sum([result["pot"]["recall"] for result in eval_results.values()]) / len(eval_results)
        avg_f1_pot = sum([result["pot"]["f1"] for result in eval_results.values()]) / len(eval_results)

        # 计算所有样本中碰撞评价指标得分
        avg_precision_collision = sum([result["collision"]["precision"] for result in eval_results.values()]) / len(eval_results)
        avg_recall_collision = sum([result["collision"]["recall"] for result in eval_results.values()]) / len(eval_results)
        avg_f1_collision = sum([result["collision"]["f1"] for result in eval_results.values()]) / len(eval_results)

        # 计算所有样本中反弹评价指标得分
        avg_precision_rebound = sum([result["rebound"]["precision"] for result in eval_results.values()]) / len(eval_results)
        avg_recall_rebound = sum([result["rebound"]["recall"] for result in eval_results.values()]) / len(eval_results)
        avg_f1_rebound = sum([result["rebound"]["f1"] for result in eval_results.values()]) / len(eval_results)

        return {
            "pot": {"precision": avg_precision_pot, "recall": avg_recall_pot, "f1": avg_f1_pot},
            "collision": {"precision": avg_precision_collision, "recall": avg_recall_collision, "f1": avg_f1_collision},
            "rebound": {"precision": avg_precision_rebound, "recall": avg_recall_rebound, "f1": avg_f1_rebound},
        }

if __name__ == '__main__':
    sam2_output_frame_dir = '/root/autodl-tmp/temp_output/det_sam2_RT_output'
    sam2_checkpoint_path = '../checkpoints/sam2.1_hiera_large.pt'
    sam2_config_path = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    detect_model_weights = 'det_weights/train_referee12_960.pt'
    visualize_result_dir = '/root/autodl-tmp/eval_output/eval_visualize'

    videos_dir = '/root/autodl-tmp/data/Det-SAM2-Evaluation/videos'  # 评估集视频文件夹
    eval_jsonl_path = '/root/autodl-tmp/data/Det-SAM2-Evaluation/postprocess.jsonl' # postprocess.jsonl 评估集标注文本jsonl文件路径
    eval_output_dir = '/root/autodl-tmp/eval_output/eval_result'  # 评估集结果输出文件夹

    det_sam2_eval = EvalDetSAM2PostProcess(
        sam2_output_frame_dir=sam2_output_frame_dir,  # 用来存储sam2的输出掩码的帧的文件夹
        sam2_checkpoint_path=sam2_checkpoint_path,  # SAM2的模型权重
        sam2_config_path=sam2_config_path,  # SAM2的配置文件
        detect_model_weights=detect_model_weights,  # 检测模型权重
        visualize_result_dir=None, # visualize_result_dir,  # 后处理可视化路径，不传即不可视化
    )

    # # 按照默认参数遍历整个评估集
    # det_sam2_eval.eval(
    #     videos_dir=videos_dir,  # 评估集视频文件夹
    #     eval_jsonl_path=eval_jsonl_path,  # 评估集标注文本jsonl文件路径
    #     eval_output_dir=eval_output_dir,  # 评估集结果输出文件夹
    # )

    # 尝试多种参数组合下，遍历整个评估集
    det_sam2_eval.eval_all_settings(
        videos_dir=videos_dir,  # 评估集视频文件夹
        eval_jsonl_path=eval_jsonl_path,  # 评估集标注文本jsonl文件路径
        eval_output_dir=eval_output_dir,  # 评估集结果输出文件夹
        detect_confidence_list=[0.7,0.8,0.9],  # 检测模型置信度阈值参数列表
        frame_buffer_size_list=[30],  # 累积帧数参数列表
        detect_interval_list=[15,30,45],  # 检测间隔参数列表
        max_frame_num_to_track_list=[60,90],  # 视频传播最大长度参数列表
        max_inference_state_frames_list=[60,120],  # 限制inference_state中保留最大帧数参数列表
        load_inference_state_path_list=[None],  # 预加载内存库路径参数列表
        pot_distance_threshold_list=[100],  # 袋口附近距离阈值参数列表
        pot_velocity_threshold_list=[0.9],  # 袋口附近速度阈值参数列表
        ball_distance_threshold_list=[120],  # 球间撞击的距离判断阈值参数列表
        ball_velocity_threshold_list=[10],  # 球间撞击的速度变化判断阈值参数列表
        table_margin_list=[100],  # 桌子边界的缓冲值参数列表
        rebound_velocity_threshold_list=[0.7],  # 判断反弹的速度阈值参数列表
    )

    print("评估完成！")
