import os
import cv2
import threading
import gc
from queue import Queue, Empty
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from readerwriterlock import rwlock
from tqdm import tqdm

from sam2.utils.misc import tensor_to_frame_rgb
from frames2video import frames_to_video
from det_sam2_RT import VideoProcessor
from postprocess_det_sam2 import VideoPostProcessor

class DetSAM2Pipeline:
    def __init__(
        self,
        # Det-SAM2实时推理主干参数
        sam2_output_frame_dir,  # SAM2渲染后的结果保存路径
        sam2_checkpoint_path,  # SAM2的模型权重
        sam2_config_path,  # SAM2的配置文件
        detect_model_weights,  # 检测模型权重
        output_video_dir, # 后处理可视化输出路径
        load_inference_state_path=None,  # 预加载内存库
        visualize_postprocessor=False, # 是否可视化后处理,如果可视化,则max_inference_state_frames不应小于总视频帧数！！
    ):
        # 初始化Det-SAM2实时推理主干的VideoProcessor类
        self.video_processor = VideoProcessor(
            output_dir=sam2_output_frame_dir,
            sam2_checkpoint=sam2_checkpoint_path,
            model_cfg=sam2_config_path,
            detect_model_weights=detect_model_weights,
            load_inference_state_path=load_inference_state_path,
            skip_classes={11, 14, 15, 19},  # 跳过的类别
            vis_frame_stride=-1, # 每隔几帧渲染一次分割结果,-1时不进行渲染
            visualize_prompt=False,  # 是否可视化交互帧
            frame_buffer_size=30,  # 每次累积多少帧后执行推理
            detect_interval=30,  # 每隔多少帧执行一次检测（条件帧提示）
            max_frame_num_to_track=60,  # 每次传播（propagate_in_video）时从最后一帧开始执行倒序追踪推理的最大长度。
            max_inference_state_frames=2000,  # 不应小于max_frame_num_to_track ; 限制inference_state中的帧数超出这个帧数时会不断释放掉旧的帧，-1表示不限制
        )
        # 初始化Det-SAM2的后处理类
        self.post_processor = VideoPostProcessor()
        # 后处理可视化输出路径
        self.visualize_postprocessor = visualize_postprocessor
        self.output_video_dir = output_video_dir

        self.inference_done_event = threading.Event()  # 用于指示推理是否完成
        self.video_segments = {}  # 中间字典，用于记录self.video_processor.video_segments中的内容，避免后处理线程与推理主干线程在self.video_processor.video_segments上冲突
        self.frames_queue = Queue()  # 使用线程安全的队列
        self.has_processed_frames = []  # 已进行后处理的帧索引列表
        # 初始化读写锁
        self.rw_lock = rwlock.RWLockWrite()
        self.post_processor_started = False  # 用于指示后处理线程是否已启动,确保后处理显存只进行一次

    def transform_video_segments(self):
        '''
        负责忠实地将SAM2推理主干中的video_segments字典内容转移到pipeline中的video_segments字典中，并为后处理线程提供帧索引和分割字典
        1.将self.video_processor.video_segments中的内容添加到self.video_segments中
        2.清空self.video_processor.video_segments以节省内存
        3.按顺序添加帧索引对应的分割字典到self.frames_queue队列
        '''
        # pre_frames = self.video_processor.pre_frames  # 预加载内存库的长度
        # need_process_frames_idx = sorted(idx for idx in self.video_processor.video_segments.keys() if idx >= pre_frames)
        need_process_frames_idx = sorted(self.video_processor.video_segments.keys())

        # 1.2.更新视频段字典并清空 video_processor 的内容
        with self.rw_lock.gen_wlock(): # 推理线程在写入时使用写锁  TODO:去掉有没有影响，可不可以去掉？
            self.video_segments.update(self.video_processor.video_segments)
            self.video_processor.video_segments.clear()

        # 3.按顺序逐一添加帧索引到 frames_queue 中
        for frame_idx in need_process_frames_idx:
            self.frames_queue.put((frame_idx, self.video_segments[frame_idx]))  # frames_queue不是对video_segments的深拷贝，而是对video_segments的引用
            # self.visualize_mask(frame_idx=frame_idx, ball_idx=9, segments=self.video_segments[frame_idx])  # debug用


    def inference(self, video_source, max_frames):
        '''
        Det-SAM2全流程推理函数，包含video_processor实时推理主干和post_processor后处理
        第一个进程：
            顺序执行 1.预加载内存库 、2.视频流主干推理 和 3.渲染分割结果
        当检测到self.video_processor.special_classes_detection有值时开始执行 4.计算后处理袋口位置和桌面边界 ，
        并开始第二个线程，异步并行，边推理边后处理：
            不断对self.video_processor.video_segments中未被后处理的帧进行 5.后处理事件监测 随后 6.释放对应self.video_segments中的帧
        第二个线程完全结束后执行6.后处理可视化
        '''
        print(f"视频源：{video_source}")

        # 第一个线程：预加载内存库和视频推理
        def process_video():
            # 1.Det-SAM2-预加载内存库
            if self.video_processor.load_inference_state_path is not None:  # 如果加载内存库路径不为空，则预加载指定内存库
                self.video_processor.inference_state = self.video_processor.load_inference_state(self.video_processor.load_inference_state_path)

                # 获取预加载内存库中条件帧和非条件帧索引，并保存记录在inference_state字典的"preloading_memory_cond_frame_idx"和"preloading_memory_non_cond_frames_idx"中
                preloading_memory_cond_frame_idx = list(self.video_processor.inference_state["output_dict"]["cond_frame_outputs"].keys())
                preloading_memory_non_cond_frames_idx = list(self.video_processor.inference_state["output_dict"]["non_cond_frame_outputs"].keys())

                self.video_processor.inference_state["preloading_memory_cond_frame_idx"] = preloading_memory_cond_frame_idx
                self.video_processor.inference_state["preloading_memory_non_cond_frames_idx"] = preloading_memory_non_cond_frames_idx
                # 获取预加载内存库的已有帧数，并更新self.pre_frames,同时保存在inference_state["preloading_memory_frames"]中
                self.video_processor.pre_frames = self.video_processor.inference_state["num_frames"]
                self.video_processor.predictor.init_preloading_state(self.video_processor.inference_state) # 将预加载内存库中部分张量移动到CPU上

            # 2.Det-SAM2-实时推理主干读取视频流（在此不支持渲染中间结果）
            if video_source is not None:
                # 从视频文件加载视频流
                cap = cv2.VideoCapture(video_source)   # TODO,是否可以控制视频流的帧率
                if not cap.isOpened():
                    print(f"无法读取视频: {video_source}")
                    return

                frame_idx = 0  # 初始化当前视频流的帧索引计数器
                while frame_idx < max_frames:
                    ret, frame = cap.read()
                    if not ret:  # 视频结束, 将缓存中累积的剩余视频帧进行推理
                        if self.video_processor.frame_buffer is not None and len(self.video_processor.frame_buffer) > 0:
                            # print(f"视频结束推理剩余帧数: {len(self.video_processor.frame_buffer)}")
                            self.video_processor.Detect_and_SAM2_inference(frame_idx=self.video_processor.pre_frames + frame_idx - 1)  # 最后一帧的索引，在循环中结束时注意需要-1
                            self.transform_video_segments()  # 将self.video_processor.video_segments中的内容添加到self.video_segments中
                        self.inference_done_event.set()  # 视频结束，标记推理完成
                        break
                    # 将BGR图像转换为RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # shape = (1080, 1920, 3)
                    self.video_processor.inference_state = self.video_processor.process_frame(self.video_processor.pre_frames + frame_idx, frame_rgb)
                    self.transform_video_segments()  # 将self.video_processor.video_segments中的内容添加到self.video_segments中

                    # 检查是否启动过后处理进程
                    if not self.post_processor_started:  # 确保后处理进程只会被启动一次
                        # 检查特殊检测的触发条件
                        if self.video_processor.special_classes_detection and not post_process_thread.is_alive():
                            # 4.Det-SAM2-后处理确定袋口位置与桌面边界
                            self.post_processor.get_hole_name(self.video_processor.special_classes_detection)  # 将袋口坐标分配給指定袋口
                            self.post_processor.get_boundary_from_holes()  # # 根据袋口坐标计算桌面有效边界
                            # 启动第二个线程（后处理）
                            post_process_thread.start()
                            self.post_processor_started = True # 标记后处理进程已启动

                    frame_idx += 1
                cap.release()

                # 在视频流中超出最大推理帧数时手动触发推理完成事件以终止后处理线程
                if frame_idx >= max_frames:
                    print(f"达到视频流处理最大帧max_frames={max_frames},终止视频流推理")
                    self.inference_done_event.set()  # 达到最大帧数时，标记推理完成

            # 3.Det-SAM2-渲染中间结果
            if self.video_processor.vis_frame_stride == -1:
                print("不对任何帧进行渲染,推理完成")
            else:
                # 首先清除self.output_dir文件夹下所有已有文件
                for file in os.listdir(self.video_processor.output_dir):
                    file_path = os.path.join(self.video_processor.output_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                # 统一渲染本次视频推理的所有帧
                images_tensor = self.video_processor.inference_state["images"]
                for i in tqdm(range(self.video_processor.pre_frames, images_tensor.shape[0]), desc="掩码可视化渲染进度"):  # 从预加载内存库后一帧开始渲染
                    # print(f"准备渲染第{i}帧记忆库图像特征")
                    if i % self.video_processor.vis_frame_stride == 0:  # i为本次推理视频帧的索引
                        tensor_img = images_tensor[i:i + 1]  # 从inferenced_state["images"]中获取self.frames，用于渲染。维度为 (1, 3, 1024, 1024)
                        frame_rgb = tensor_to_frame_rgb(tensor_img)  # 当前RGB帧的nd数组
                        self.video_processor.render_frame(i, frame_rgb, self.video_segments)  # 根据结果字典渲染所需要的时包含预加载内存库的绝对索引，因此i+self.pre_frames
                print(f"---按照每{self.video_processor.vis_frame_stride}帧间隔渲染,渲染结束")
                frames_to_video(
                    frames_folder=self.video_processor.output_dir,
                    output_video_path='./temp_output/output_segment_video.mp4',
                    fps=2
                )

        # 第二个线程：后处理
        def post_process():
            '''
            后处理线程，不断对video_segments中未被后处理的帧进行后处理（确保后处理按索引帧进行顺序处理新的帧，同时需要处理SAM2推理主干中第二次处理的修正帧）
            如果推理完成且待处理帧队列为空，则结束线程
            '''
            while True:
                try:
                    # 如果推理完成且没有未处理的帧，则退出线程
                    if self.inference_done_event.is_set() and self.frames_queue.empty():
                        break

                    frame_idx, segments = self.frames_queue.get()  # 从队列中获取并删除一个帧索引与其对应的分割结果
                    frame_idx = frame_idx - self.video_processor.pre_frames  # 修正帧索引，因为后处理线程只处理推理主干中非预加载内存库部分
                    # print("后处理接收帧",frame_idx)
                    # 特别注意：推理主干SAM2实际上是每一帧处理两遍以具备修正能力的（propograte长度两倍于视频累积缓存长度），因此后处理也需要具备处理重复已处理帧的能力，同时还需要处理未处理的帧时按顺序进行不能跳过
                    if frame_idx <= len(self.has_processed_frames):  # 可以处理过去处理过的帧，但不允许跳过一些帧直接处理后面的

                        # 5.Det-SAM2-后处理：对每一帧进行事件监测
                        # 计算当前帧中所有球的位置
                        self.post_processor.balls_positions[frame_idx] = self.post_processor.process_frame_positions(segments)  # TODO:给定的MASK有问题，计算出错误的坐标
                        # self.visualize_mask(frame_idx=frame_idx, ball_idx=9, segments=segments)  # debug用

                        if frame_idx > 0:  # 从第二帧开始才有可能计算速度向量
                            # 计算当前帧所有球的速度向量
                            self.post_processor.balls_velocities[frame_idx] = self.post_processor.process_frame_velocities(frame_idx, time_interval=1.0)
                            # 检查这一帧是否有球进洞
                            self.post_processor.check_ball_disappeared_pot(frame_idx)
                            if frame_idx > 1:  # 要计算速度向量的变化差值需要比计算速度向量后一帧
                                # 检查这一帧是否有球间碰撞
                                self.post_processor.check_ball_collision(frame_idx)
                                # 检查这一帧是否有球在桌边反弹
                                self.post_processor.check_ball_rebound(frame_idx)
                        # 标记已处理的帧
                        if frame_idx not in self.has_processed_frames:
                            # print(f"后处理新处理{frame_idx}帧，并标记")
                            self.has_processed_frames.append(frame_idx)
                        # else:
                        #     print(f"后处理重复处理{frame_idx}帧")

                        # 6.不可视化分割结果时释放对应self.video_segments中的帧
                        if self.video_processor.vis_frame_stride == -1:
                            self.video_segments.pop(frame_idx, None)
                            gc.collect()

                except Empty:
                    continue

            # 6.Det-SAM2-后处理可视化
            if self.visualize_postprocessor:
                # 从video_processor.inference_state["images"]中获取张量处理成RGB帧的nd数组的列表
                frames_rgb = []
                images_tensor = self.video_processor.inference_state["images"]
                for i in range(self.video_processor.pre_frames, images_tensor.shape[0]):
                    tensor_img = images_tensor[i:i + 1]  # 维度为 (1, 3, 1024, 1024)
                    frame_rgb = tensor_to_frame_rgb(tensor_img)
                    frames_rgb.append(frame_rgb)
                self.post_processor.visualize(
                    video_source=frames_rgb,  # 传入RGB帧的nd数组的列表
                    output_video_dir=self.output_video_dir,
                )

            # 打印结果字典
            # print(f"进洞信息:{self.post_processor.disappeared_balls}")
            # print(f"碰撞信息:{self.post_processor.ball_collision}")
            # print(f"反弹信息:{self.post_processor.ball_rebound}")

        # 启动第一个线程（视频推理）
        video_thread = threading.Thread(target=process_video)
        video_thread.start()

        # 启动第二个线程（后处理），但只有在特殊检测触发后才真正开始处理
        post_process_thread = threading.Thread(target=post_process)

    def visualize_mask(self,frame_idx, ball_idx, segments):
        '''
        可视化分割mask中特定类别的情况，主要用于开发时debug
        '''
        mask = segments[ball_idx]
        # 将掩膜从 (1, 1080, 1920) 转换为 (1080, 1920)
        if len(mask.shape) != 2:
            mask = np.squeeze(mask, axis=0)
        # 将布尔掩码转换为 0 和 255 的图像格式
        mask_img = np.uint8(mask * 255)
        # 使用 Matplotlib 可视化
        plt.imshow(mask_img, cmap='gray')
        plt.title(f"Frame {frame_idx} - Ball {ball_idx} Mask")
        plt.axis('off')
        # plt.show()

        # 保存图像
        file_path = f"debug/frame_{frame_idx:05d}.png"
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    sam2_output_frame_dir = './temp_output/video_frames' # '/root/autodl-tmp/temp_output/video_frames'
    sam2_checkpoint_path = '../checkpoints/sam2.1_hiera_large.pt'
    sam2_config_path = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    detect_model_weights = 'det_weights/train_referee12_960.pt'
    load_inference_state_path = '/root/autodl-tmp/output_inference_state/inference_state_10frames.pkl'
    output_video_dir = './pipeline_output' # '/root/autodl-tmp/pipeline_output'
    # 视频源，可以是本地视频文件或者rtsp流
    video_path = './data/Det-SAM2-Evaluation/videos/video5.mp4'
    # video_path = './data/LongVideo/5min.mp4'  # Det-SAM2评估集/videos/video5.mp4'
    rtsp_url = 'rtsp://175.178.18.243:19699/'

    # 初始化Det-SAM2全流程pipeline类
    pipeline = DetSAM2Pipeline(
        sam2_output_frame_dir=sam2_output_frame_dir,
        sam2_checkpoint_path=sam2_checkpoint_path,
        sam2_config_path=sam2_config_path,
        detect_model_weights=detect_model_weights,
        # load_inference_state_path=load_inference_state_path,
        output_video_dir=output_video_dir,
        visualize_postprocessor=True,  # 是否可视化后处理,如果可视化,则max_inference_state_frames不应小于总视频帧数！！
    )
    # 执行全流程推理
    pipeline.inference(video_source=rtsp_url, max_frames=300)

