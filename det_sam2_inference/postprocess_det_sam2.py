import os
import cv2
import numpy as np
import pickle

from sympy import andre
from tqdm import tqdm

class VideoPostProcessor:
    def __init__(
            self,
            pot_distance_threshold=100,
            pot_velocity_threshold=0.9,
            ball_distance_threshold=120, # 球与球之间撞击的距离判断阈值
            ball_velocity_threshold=10, # 球与球之间撞击的速度变化判断阈值
            table_margin=100,  # 桌子边界的缓冲值,增大可以增加判断球是否在桌边反弹的范围
            rebound_velocity_threshold=0.7,  # 判断反弹的速度阈值,增大可以则更容易被判断成反弹（表示碰撞前后垂直速度分量大小的变化不超过这个阈值）
    ):
        self.hole_names_and_positions = []  # 存储球桌每个洞口的坐标和其对应的洞口名称 # [(name,positions),(name,positions),...]
        self.effective_boundary = None  # 由洞口坐标计算出的有效的桌面边界
        self.balls_positions = {}  # 记录每一帧中每个球的位置
        self.balls_velocities = {}  # 记录每一帧中每个球的速度向量

        self.disappeared_balls = {}  # 记录消失的球及其最后出现的帧索引和位置  # self.disappeared_balls[ball_id] = {"last_frame": frame_idx, "last_position": last_position}
        self.ball_collision = {}  # 记录每一帧中发生的球与球碰撞的信息  # self.ball_collision[frame_idx] = [(ball_id1,ball_id2),(ball_id3,ball_id4)...]
        self.ball_rebound = {}  # 记录每一帧中球撞击边界反弹的信息 # self.ball_rebound[frame_idx] = [(ball_id1,"top"), (ball_id2,"right"),...]

        # 判断接近洞口的距离阈值和速度指向洞口的阈值
        self.pot_distance_threshold = pot_distance_threshold  # 增大可以提供更大的判定范围
        self.pot_velocity_threshold = pot_velocity_threshold  # 速度向量的方向阈值，增大可以更宽松地允许球速度方向偏离洞口
        # 判断球与球之间的碰撞的阈值
        self.ball_distance_threshold = ball_distance_threshold  # 球与球之间撞击的距离判断阈值
        self.ball_velocity_threshold = ball_velocity_threshold  # 球与球之间撞击的速度变化判断阈值
        # 判断球是否在桌边反弹的判定区域阈值
        self.margin = table_margin  # 桌子边界的缓冲值，增大可以提供更大的判定范围
        # 判断球是否在桌边反弹的速度变化阈值
        self.rebound_velocity_threshold = rebound_velocity_threshold  # 判断反弹的阈值，增大可以提供更大的判定范围。表示碰撞前后垂直速度分量大小的变化不超过这个阈值

    def clear(self):
        """
        重置后处理所有状态，以接收新的视频
        """
        self.hole_names_and_positions = []  # 重置洞口信息列表
        self.effective_boundary = None  # 重置有效边界
        self.balls_positions = {}  # 重置位置字典
        self.balls_velocities = {}  # 重置速度字典

        self.disappeared_balls = {}  # 重置进球字典
        self.ball_collision = {}  # 重置碰撞字典
        self.ball_rebound = {}  # 重置反弹字典


    def load_video_segments(self, file_path):
        """从pkl中读取video_segments字典"""
        with open(file_path, 'rb') as file:
            video_segments = pickle.load(file)
        print(f"---从{file_path}加载video_segments")
        return video_segments

    # （下）一些可视化方法
    def visualize(self, video_source, output_video_dir, output_video_name="postprocess_visualized.mp4"):
        """
        video_source 目前只支持读取本地视频路径,或以RGB帧的np数组形式存储每一帧的视频的列表
        在视频的每一帧上可视化袋口中心点、其判定范围以及球的速度向量
        """

        output_video_path = os.path.join(output_video_dir, output_video_name)

        if isinstance(video_source, str) and os.path.isfile(video_source): # 如果视频源是本地视频路径
            cap = cv2.VideoCapture(video_source)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 2 # int(cap.get(cv2.CAP_PROP_FPS))  # 设置一秒2帧的输出视频
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 设置输出视频编码和参数
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        elif isinstance(video_source[0], np.ndarray):  # 如果视频源是RGB帧的np数组组成的列表
            total_frames = len(video_source)
            width, height = video_source[0].shape[1], video_source[0].shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 2, (width, height))

        else:
            print(f"---{video_source}既不是本地视频，也不是存储RGB帧的np数组列表")

        # 使用 tqdm 显示处理进度条
        for frame_idx in tqdm(range(total_frames), desc="后处理可视化渲染"):
            if isinstance(video_source, str) and os.path.isfile(video_source):  # 如果视频源是本地视频路径
                ret, frame = cap.read()
                if not ret:
                    break
            else:  # 如果视频源是RGB帧的np数组组成的列表
                frame = video_source[frame_idx]
                # 如果是RGB格式的帧，转换为BGR以便保存到视频文件
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 在当前帧上绘制袋口中心点及其判定范围
            for hole_name, hole_center in self.hole_names_and_positions:
                center = tuple(map(int, hole_center))
                # 绘制袋口中心点
                cv2.circle(frame, center, 10, (0, 0, 255), -1)  # 红色实心圆表示袋口
                # 绘制判定范围圆圈
                cv2.circle(frame, center, self.pot_distance_threshold, (0, 255, 0), 2)  # 绿色圆圈表示距离阈值
                # 在袋口附近标注袋口名称
                cv2.putText(frame, hole_name, (center[0] + 15, center[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 在当前帧上绘制球的位置和速度向量
            current_ball_positions = self.balls_positions.get(frame_idx, {})
            current_ball_velocities = self.balls_velocities.get(frame_idx, {})
            for ball_id, position in current_ball_positions.items():
                if position is not None:
                    # 绘制球的位置
                    cv2.circle(frame, position, 8, (0, 0, 255), -1)  # 黄色实心圆表示球
                    # 绘制速度向量
                    velocity = current_ball_velocities.get(ball_id, (0, 0))
                    end_point = (int(position[0] + velocity[0]), int(position[1] + velocity[1]))
                    cv2.arrowedLine(frame, position, end_point, (0, 0, 255), 4, tipLength=0.1)  # 黄色箭头表示速度向量
                    # 在球旁边显示球的ID
                    cv2.putText(frame, str(ball_id), (position[0] + 10, position[1] - 10),  # 在球上方显示ID
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示进洞信息
            for ball_id, info in self.disappeared_balls.items():
                if frame_idx >= info["last_frame"] and frame_idx <= info["last_frame"] + 10:
                    x, y = info["last_position"]  # 获取最后位置
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 3)  # 黄色边框表示进洞

                    # 在球的右侧显示球ID和进洞信息
                    text = f"{ball_id} In {info['hole']}"
                    text_x = int(x) + 10  # 在球的右侧显示，向右偏移10个像素
                    text_y = int(y)  # 垂直对齐到球的位置
                    cv2.putText(frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示碰撞信息
            collision_text = ""  # 初始化变量
            if frame_idx in self.ball_collision:
                # 获取当前帧的碰撞信息
                current_collisions = self.ball_collision[frame_idx]
                if current_collisions:
                    collision_info = [f"Ball {ball_id1} & Ball {ball_id2}" for ball_id1, ball_id2 in current_collisions]
                    collision_text = f"{frame_idx} frame collisions: " + ", ".join(collision_info)
                    # 在碰撞球周围绘制圆圈
                    for ball_id1, ball_id2 in current_collisions:
                        pos1 = self.balls_positions.get(frame_idx, {}).get(ball_id1)
                        pos2 = self.balls_positions.get(frame_idx, {}).get(ball_id2)
                        if pos1 is not None:
                            cv2.circle(frame, tuple(map(int, pos1)), 25, (0, 0, 255), 3)  # 红色圆圈表示球1
                        if pos2 is not None:
                            cv2.circle(frame, tuple(map(int, pos2)), 25, (0, 0, 255), 3)  # 红色圆圈表示球2
            # 如果碰撞信息存在，则绘制
            if collision_text:
                text_size = cv2.getTextSize(collision_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]  # 字体大小增大
                # 计算文本在视频底部的位置
                text_x = (width - text_size[0]) // 2  # 水平居中
                text_y = height - 10  # 垂直位置距底部10个像素
                # 在当前帧上绘制碰撞信息
                cv2.putText(frame, collision_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 字体大小和粗细增加

            # 在当前帧上绘制有效桌面边界
            left_buffer, right_buffer, top_buffer, bottom_buffer = self.effective_boundary
            left = left_buffer - self.margin
            right = right_buffer + self.margin
            top = top_buffer - self.margin
            bottom = bottom_buffer + self.margin
            # 绿色线表示有效的边界
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(left_buffer), int(top_buffer)), (int(right_buffer), int(bottom_buffer)), (0, 255, 0), 2)  # 黄色线表示有效的边界
            # 获取当前帧的反弹信息
            rebound = self.ball_rebound.get(frame_idx, [])
            # 如果有反弹球，更新边界线的颜色并显示球的ID
            if rebound:
                # 在边界中间显示反弹球的ID
                for ball_id, direction in rebound:
                    if direction == "top":
                        # 如果反弹在顶部，将上边界线变为红色
                        cv2.line(frame, (int(left), int(top)), (int(right), int(top)), (0, 0, 255), 2)  # 顶部边界
                        cv2.line(frame, (int(left_buffer), int(top_buffer)), (int(right_buffer), int(top_buffer)),
                                 (0, 0, 255), 2)  # 内部顶部边界
                        # 在内部顶部边界中间显示球的ID
                        text_x = (left_buffer + right_buffer) // 2
                        text_y = top_buffer + 20  # 在边界上方显示
                        cv2.putText(frame, str(ball_id), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (255, 255, 255), 3)
                    elif direction == "bottom":
                        # 如果反弹在底部，将下边界线变为红色
                        cv2.line(frame, (int(left), int(bottom)), (int(right), int(bottom)), (0, 0, 255), 2)  # 底部边界
                        cv2.line(frame, (int(left_buffer), int(bottom_buffer)), (int(right_buffer), int(bottom_buffer)),
                                 (0, 0, 255), 2)  # 内部底部边界
                        # 在内部底部边界中间显示球的ID
                        text_x = (left_buffer + right_buffer) // 2
                        text_y = bottom_buffer - 10  # 在边界下方显示
                        cv2.putText(frame, str(ball_id), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (255, 255, 255), 3)
                    elif direction == "left":
                        # 如果反弹在左侧，将左边界线变为红色
                        cv2.line(frame, (int(left), int(top)), (int(left), int(bottom)), (0, 0, 255), 2)  # 左边界
                        cv2.line(frame, (int(left_buffer), int(top_buffer)), (int(left_buffer), int(bottom_buffer)),
                                 (0, 0, 255), 2)  # 内部左边界
                        # 在内部左边界中间显示球的ID
                        text_x = left_buffer + 10  # 在边界左侧显示
                        text_y = (top_buffer + bottom_buffer) // 2
                        cv2.putText(frame, str(ball_id), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (255, 255, 255), 3)
                    elif direction == "right":
                        # 如果反弹在右侧，将右边界线变为红色
                        cv2.line(frame, (int(right), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)  # 右边界
                        cv2.line(frame, (int(right_buffer), int(top_buffer)), (int(right_buffer), int(bottom_buffer)),
                                 (0, 0, 255), 2)  # 内部右边界
                        # 在内部右边界中间显示球的ID
                        text_x = right_buffer - 50  # 在边界右侧显示
                        text_y = (top_buffer + bottom_buffer) // 2
                        cv2.putText(frame, str(ball_id), (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (255, 255, 255), 3)

                # 显示当前帧数在左上角
            frame_count_text = f"Frame: {frame_idx + 1}/{total_frames}"
            cv2.putText(frame, frame_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 将处理后的帧写入输出视频
            out.write(frame)

        # 释放资源
        if isinstance(video_source, str) and os.path.isfile(video_source):
            cap.release()
        out.release()
        print(f"---视频已保存至 {output_video_path}")


    # （下）将袋口坐标分配給指定袋口，并计算有效桌面边界 -----------------------------------------------------------------------------------------

    def get_hole_name(self, poket_pkl):
        """根据洞口坐标获取洞口名称"""
        # 假设的袋口位置
        holes_positions = {
            "left_up": (100, 100),"middle_up": (960, 0),"right_up": (1820, 100),"left_down": (100, 720),"middle_down": (960, 720),"right_down": (1820, 720),
        }

        # 如果poket_pkl是文件路径
        if isinstance(poket_pkl, str):
            with open(poket_pkl, 'rb') as file:
                coordinates = pickle.load(file)
            if coordinates is None:
                print(f"---没有获取到袋口信息")
        # 如果poket_pkl是字典
        else:
            coordinates = poket_pkl


        # 分析坐标
        for coord in coordinates:
            # 计算中心点坐标
            center_x = (coord[0] + coord[2]) / 2
            center_y = (coord[1] + coord[3]) / 2
            center = (center_x, center_y)

            closest_hole = None
            min_distance = float('inf')

            # 计算与每个洞口的距离
            for hole_name, hole_pos in holes_positions.items():
                distance = np.linalg.norm(np.array(center) - np.array(hole_pos))

                if distance < min_distance:
                    min_distance = distance
                    closest_hole = hole_name

            # 只记录非重复的洞口名称和对应坐标
            if closest_hole is not None:
                self.hole_names_and_positions.append((closest_hole, center))

    def get_boundary_from_holes(self):
        """根据洞口坐标获取有效的撞击边界"""
        if not self.hole_names_and_positions:
            raise ValueError("---No hole positions available to define boundaries.")

        # 假设按照 "left_up", "right_up", "left_down", "right_down"
        positions = {name: pos for name, pos in self.hole_names_and_positions}
        try:
            left_up = positions["left_up"]
            right_up = positions["right_up"]
            left_down = positions["left_down"]
            right_down = positions["right_down"]
        except KeyError as e:
            print(f"---缺少关键四角洞口坐标：{e}")

        # 计算有效桌面边界
        left = min(left_up[0], left_down[0]) + self.margin
        right = max(right_up[0], right_down[0]) - self.margin
        top = min(left_up[1], right_up[1]) + self.margin
        bottom = max(left_down[1], right_down[1]) - self.margin

        self.effective_boundary = (left, right, top, bottom)

    # （下）获取球的质心坐标和速度向量 ---------------------------------------------------------------------------------------

    def remove_white_ball_from_other_masks(self, white_ball_mask, other_ball_masks, dilation_iterations=1):
        """
        从别的球的掩码中减去白球的掩码,别的球掩码与白球掩码重叠时不一定完美重合，
        因此对白球掩码先进行膨胀操作来增加别的球掩码被完全修正的概率
        :param white_ball_mask: 白球的二进制掩码 (NumPy array, bool or uint8)
        :param other_ball_masks: 别的球的二进制掩码列表 (每个元素都是 NumPy array, bool or uint8)
        :return: 更新后的别的球掩码列表
        """
        white_ball_mask = np.squeeze(white_ball_mask, axis=0)
        # 确保 white_ball_mask 是布尔类型
        white_ball_mask = white_ball_mask.astype(bool).astype(np.uint8)

        # 创建膨胀所需的内核，kernel_size 可以根据需要调整
        kernel = np.ones((3, 3), np.uint8)
        # 对白球掩码进行膨胀
        dilated_white_ball_mask = cv2.dilate(white_ball_mask, kernel, iterations=dilation_iterations)

        updated_masks = []
        for mask in other_ball_masks:
            mask = np.squeeze(mask, axis=0)
            # 确保其他球的掩码也是布尔类型
            mask = mask.astype(bool)
            # 从其他球掩码中移除与白球掩码重叠部分
            # updated_mask = np.where(white_ball_mask, 0, mask).astype(np.uint8)
            updated_mask = cv2.bitwise_and(mask.astype(np.uint8), cv2.bitwise_not(dilated_white_ball_mask))
            updated_masks.append(updated_mask)

        return updated_masks

    def get_position(self, mask):
        """从掩膜中计算球的质心坐标"""
        # 将掩膜从 (1, 1080, 1920) 转换为 (1080, 1920)
        if len(mask.shape) != 2:
            mask = np.squeeze(mask, axis=0)
        mask_uint8 = mask.astype(np.uint8) * 255  # 将 video_segment 的 bool 转换为 uint8
        moments = cv2.moments(mask_uint8)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)  # 质心坐标
        else:
            return None  # 如果掩膜面积为 0，则没有质心

    def process_frame_positions(self, frame_segments):
        """计算每一帧中每个球的质心位置"""
        current_positions = {}
        # 获取白球的掩码
        white_ball_mask = frame_segments.get(16, None)

        # 遍历这一帧中所有的球
        for ball_id, mask in frame_segments.items():
            if ball_id != 16:
                # 修正可能出现的掩码与白球掩码重叠的情况，将掩码减去白球掩码
                mask = self.remove_white_ball_from_other_masks(white_ball_mask, [mask])[0]

            # 获取当前球的位置
            current_position = self.get_position(mask)
            current_positions[ball_id] = current_position
        return current_positions

    def get_velocity(self, pos1, pos2, time_interval=1.0):
        """根据两帧的位置信息计算速度向量"""
        if pos1 is None or pos2 is None:
            return (0, 0)  # 如果位置不可用，返回零向量
        vx = (pos2[0] - pos1[0]) / time_interval
        vy = (pos2[1] - pos1[1]) / time_interval
        return (vx, vy)

    def process_frame_velocities(self, frame_idx, time_interval=1.0, max_backtrack=5):
        """
        计算速度向量，正常情况根据当前帧和上一帧的位置信息
        然而位置坐标可能不连续
        :param frame_idx: 当前帧的索引
        :param time_interval: 帧之间的时间间隔（假设帧率固定）
        :param max_backtrack: 最大回溯帧数，以寻找缺失位置的前一有效帧
        """
        current_velocities = {}
        current_positions = self.balls_positions[frame_idx] # 当前位置信息

        for ball_id, current_position in current_positions.items():
            previous_position = None
            effective_time_interval = time_interval

            # 回溯找到上一帧有效的位置信息
            for backtrack in range(1, max_backtrack + 1):
                prev_frame_idx = frame_idx - backtrack
                if prev_frame_idx in self.balls_positions:
                    previous_position = self.balls_positions[prev_frame_idx].get(ball_id, None)
                    if previous_position is not None:
                        # 找到前一帧有效的位置信息，更新有效时间间隔
                        effective_time_interval = time_interval * backtrack
                        break

            # 如果没有找到有效的前一帧位置，返回速度为 (0, 0)
            if previous_position is None:
                velocity = (0, 0)
            else:
                # 计算速度
                velocity = self.get_velocity(previous_position, current_position, effective_time_interval)
            current_velocities[ball_id] = velocity
        return current_velocities

    # （下）检测球进洞 ----------------------------------------------------------------------------------------------------

    def check_ball_disappeared_pot(self, frame_idx):
        """
        检查球在洞口附近消失并记录其最后信息：
        球在上一帧的位置在洞口附近，且速度指向洞口，且之后不再出现

        对帧重复进行多次后处理以修正时，每个球只有一个字典元素，只有一次进球信息，最终进球信息会覆盖，无需手动判断覆盖
        """
        current_positions = self.balls_positions[frame_idx]
        previous_positions = self.balls_positions[frame_idx - 1]

        for ball_id, prev_position in previous_positions.items():
            current_position = current_positions.get(ball_id, None)
            if current_position is None:  # 如果在当前帧球的坐标为 None，说明消失了
                # 遍历每个球洞
                for hole_name, hole_position in self.hole_names_and_positions:
                    # 检查该球某一帧是否接近洞口，并且速度指向洞口
                    is_near_hole, distance = self.is_near_hole(prev_position, hole_position)
                    # print(f"第{frame_idx}帧：球{ball_id}距离{hole_name}洞口的距离为{distance}")
                    if is_near_hole:
                        # print(f"进洞监测：{frame_idx}帧-球{ball_id}在{hole_name}附近")
                        if self.is_velocity_towards_hole(ball_id, prev_position, frame_idx):
                            # print(f"进洞监测：{frame_idx}帧-球{ball_id}速度指向{hole_name}")
                            # 记录最新可能进洞的球的信息（或覆盖此前的信息）
                            self.disappeared_balls[ball_id] = {
                                "last_frame": frame_idx-1,
                                "last_position": prev_position,
                                "hole": hole_name  # 记录进入的洞口名称
                            }
                            print(f"---进洞监测：{frame_idx}帧-球{ball_id}在洞口{hole_name}消失-最后位置{prev_position}")


    def is_near_hole(self, position, hole_position):
        """检查球是否接近洞口"""
        if position is None:
            return False, None  # 如果位置为 None，直接返回 False
        # print(f"球位置：{position}，洞口位置：{hole_position}")
        distance = np.linalg.norm(np.array(position) - np.array(hole_position))
        is_near_hole = distance < self.pot_distance_threshold  # 距离阈值 self.distance_threshold
        return is_near_hole, distance

    def is_velocity_towards_hole(self, ball_id, position, frame_idx):
        """检查球的速度是否指向洞口"""
        last_velocity = self.balls_velocities[frame_idx-1].get(ball_id)
        if last_velocity:
            # 计算每个洞口相对于球当前坐标的向量
            hole_vectors = [np.array(hole[1]) - np.array(position) for hole in self.hole_names_and_positions]
            for hole_vector in hole_vectors:
                # 计算洞口向量的单位方向
                hole_direction = hole_vector / np.linalg.norm(hole_vector)
                # 计算速度向量的单位方向
                velocity_direction = np.array(last_velocity) / np.linalg.norm(last_velocity)
                # 计算洞口方向与速度方向的点积
                # 如果点积大于self.velocity_threshold，说明速度方向大致指向洞口
                if np.dot(hole_direction, velocity_direction) > self.pot_velocity_threshold:
                    return True  # 球的速度指向该洞口

        return False  # 球的速度没有指向任何洞口

    # （下）检测球与球之间的碰撞 --------------------------------------------------------------------------------------------

    def check_ball_collision(self, frame_idx):
        """
        当球速度向量发生明显变化的时候开始碰撞检测判断：
        1.找到可能与当前球发生碰撞——符合碰撞后后变化的速度向量（碰撞方向的速度分量将被反转，非碰撞方向的速度分量保持不变）
        2.判断可能发生碰撞的球是否在当前球附近

        对帧重复进行多次后处理以修正时,碰撞信息会进行覆盖，不需要手动清除此前可能发生的错误信息
        """
        current_frame_collisions = []
        # 遍历所有球，检查是否发生显著的速度变化
        for ball_id, velocity in self.balls_velocities[frame_idx].items():
            prev_velocity = self.balls_velocities[frame_idx - 1].get(ball_id, (0, 0))
            # 计算速度变化，判断是否有明显变化
            velocity_change = self.get_velocity_change(velocity, prev_velocity)
            # print(f"{frame_idx}帧中球{ball_id}的速度变化为{velocity_change}")
            if velocity_change > self.ball_velocity_threshold:
                # print(f"碰撞监测：{frame_idx}帧-球{ball_id}速度异常变化{velocity_change}")

                # 找到可能与之发生碰撞的球的列表
                potential_collisions = self.find_potential_collisions(ball_id, frame_idx)
                for other_ball_id in potential_collisions:
                    print(f"---碰撞监测：{frame_idx}帧-球{ball_id}与球{other_ball_id}可能发生碰撞")
                    current_frame_collisions.append((ball_id, other_ball_id))

        # 碰撞信息保存至self.ball_collision
        self.ball_collision[frame_idx] = current_frame_collisions

    def get_velocity_change(self, velocity, prev_velocity):
        """计算速度变化的大小，标量"""
        vx_change = velocity[0] - prev_velocity[0]
        vy_change = velocity[1] - prev_velocity[1]
        return (vx_change ** 2 + vy_change ** 2) ** 0.5  # 速度变化的大小（欧氏距离）

    def find_potential_collisions(self, ball_id, frame_idx):
        """
        寻找可能与给定球发生碰撞的其他球:
        - 如果球之间的距离小于设定的距离阈值，则可能发生碰撞
        - 且球与给定球之间的速度向量变化符合碰撞规律
        """
        potential_collisions = []

        # 获取给定球前一帧速度向量和位置坐标，获取当前帧给定球的速度向量和位置坐标
        prev_position = self.balls_positions[frame_idx-1].get(ball_id)
        current_position = self.balls_positions[frame_idx].get(ball_id)
        prev_velocity = self.balls_velocities[frame_idx-1].get(ball_id)
        current_velocity = self.balls_velocities[frame_idx].get(ball_id)

        # 如果当前球的位置为 None，无法计算距离，跳过
        if current_position is None:
            return potential_collisions

        # 遍历所有球，检查是否与当前球发生碰撞
        for other_ball_id, current_other_position in self.balls_positions[frame_idx].items():
            if other_ball_id != ball_id and current_other_position is not None:
                # 计算两个球的距离
                distance = np.linalg.norm(np.array(current_position) - np.array(current_other_position))
                # print(f"判断{ball_id}和{other_ball_id}的距离为{distance}")
                # 如果该球与当前球的距离小于设定的可能碰撞的距离阈值，则进行进一步速度向量的检查
                if distance < self.ball_distance_threshold:
                    # 获取另一个球前一帧速度向量和位置坐标，获取另一个球当前帧的速度向量
                    prev_other_position = self.balls_positions[frame_idx-1].get(other_ball_id)
                    current_other_velocity = self.balls_velocities[frame_idx].get(other_ball_id)
                    prev_other_velocity = self.balls_velocities[frame_idx-1].get(other_ball_id)
                    # print(f"判断{ball_id}和{other_ball_id}是否发生碰撞")
                    # 根据速度变化判断两个球是否发生了碰撞
                    if self.is_valid_collision(
                            prev_velocity1=prev_velocity,  # 前一帧球1的速度向量
                            prev_velocity2=prev_other_velocity,  # 前一帧球2的速度向量
                            prev_position1=prev_position,  # 前一帧球1的位置坐标
                            prev_position2=prev_other_position,  # 前一帧球2的位置坐标
                            curr_velocity1=current_velocity,  # 当前帧球1的速度向量
                            curr_velocity2=current_other_velocity,  # 当前帧球2的速度向量
                    ):
                        potential_collisions.append(other_ball_id)

        return potential_collisions

    def is_valid_collision(
            self,
            prev_velocity1,  # 前一帧球1的速度向量
            prev_velocity2,  # 前一帧球2的速度向量
            prev_position1,  # 前一帧球1的位置坐标
            prev_position2,  # 前一帧球2的位置坐标
            curr_velocity1,  # 当前帧球1的速度向量
            curr_velocity2,  # 当前帧球2的速度向量
    ):
        """
        判断两个球是否发生了有效碰撞:
        - 碰撞前两球相向运动
        - 碰撞后两球的速度发生明显变化应该有相关性,例如新增相互远离的分量
        """
        # 判断碰撞前一帧两球是否相向运动
        if self.is_moving_towards(prev_velocity1, prev_velocity2, prev_position1, prev_position2):
            # 计算碰撞前的相对速度
            prev_relative_velocity = np.array(prev_velocity1) - np.array(prev_velocity2)
            # 计算碰撞后的相对速度
            curr_relative_velocity = np.array(curr_velocity1) - np.array(curr_velocity2)

            # 判断碰撞后两球的相对速度是否相互远离 TODO：是否能够判断两个球碰撞后都朝着同一个方向移动的情况？
            # 通过相对速度的点积来判断，如果点积为正，说明两球相互远离
            # print(np.dot(prev_relative_velocity, curr_relative_velocity))
            if np.dot(prev_relative_velocity, curr_relative_velocity) < 0:
                return True

            # # 计算碰撞后两球的速度变化具有相关性
            # velocity_change1 = np.array(curr_velocity1) - np.array(prev_velocity1)
            # velocity_change2 = np.array(curr_velocity2) - np.array(prev_velocity2)
            # dot_product = np.dot(velocity_change1, velocity_change2)
            # if dot_product < 0:  # 速度变化方向相反表示碰撞
            #     return True

        return False

    def is_moving_towards(self,velocity1, velocity2, position1, position2):
        """判断两个球是否相向运动"""
        # 如果该球位置信息不存在，则不可能出现相向运动，返回False
        if position1 is None or position2 is None:
            return False
        if velocity1 is None or velocity2 is None:
            return False

        # 两个球的速度向量相减可得相对速度
        relative_velocity = np.array(velocity1) - np.array(velocity2)
        # 计算两个球的位置差向量（即两球之间的连线方向）
        position_diff = np.array(position1) - np.array(position2)

        # 计算相对速度和位置差向量的点积，判断它们是否方向一致
        dot_product = np.dot(relative_velocity, position_diff)
        # 如果点积为负，说明相对速度的方向与球之间的连线方向一致，即两个球正在相互靠近
        if dot_product < 0 :
            return True
        return False

    # （下）检测球是否在桌边反弹 --------------------------------------------------------------------------------------------

    def check_ball_rebound(self, frame_idx):
        """
        优化后的反弹逻辑：
        1. 检查当前位置是否处于缓冲边界以内，如果是则记录缓冲边界的位置（上、下、左、右）
        2. 判断反弹前球是否与对应边界相向运动
        3. 判断反弹后球是否与边界相离运动，
        4. 检查垂直于边界的速度分量的变化是否符合反弹规律，如果不符合，继续检查平行于边界的速度分量是否基本一致
        5. 如果满足1,2不满足3,4，则继续检查是否因为撞击的是袋口附近的弧面而无法用一般的规律来判断（判断是否位于袋口附近）

        对帧重复进行多次后处理以修正时,反弹信息会进行覆盖,不需要手动清除此前可能发生的错误信息
        """
        # 获取当前帧的位置和速度向量，前一帧的位置和速度向量
        current_positions = self.balls_positions[frame_idx]
        previous_positions = self.balls_positions[frame_idx - 1]
        velocities = self.balls_velocities[frame_idx]
        previous_velocities = self.balls_velocities[frame_idx - 1]

        rebounded_balls = []  # 存储反弹的球

        # 遍历当前帧每一颗球的位置
        for ball_id, current_pos in current_positions.items():
            previous_pos = previous_positions.get(ball_id)  # 获取球的前一位置
            velocity = velocities.get(ball_id)  # 获取球的速度向量
            previous_velocity = previous_velocities.get(ball_id)  # 前一帧的速度向量

            # 确保位置和速度不为 None
            if current_pos is not None and previous_pos is not None and velocity is not None:
                # 获取当前帧的xy坐标，获取当前帧的速度向量中x和y的分量
                prev_x, prev_y = previous_pos
                curr_x, curr_y = current_pos
                vel_x, vel_y = velocity
                prev_vel_x, prev_vel_y = previous_velocity

                # 记录当前帧是否可能触碰到的边界反弹
                touched_boundary = None
                # 检查当前帧是否在缓冲区域内
                buffer_zone_current = self.is_in_buffer_zone(curr_x, curr_y)
                # 检查前一帧是否在缓冲区域内
                buffer_zone_previous = self.is_in_buffer_zone(prev_x, prev_y)

                if buffer_zone_current is not None and buffer_zone_previous is not None:
                    # print(f"反弹监测：{frame_idx}帧-{ball_id}球符合位置条件")
                    # 如果前一帧和当前帧都在缓冲区域内，记录缓冲区域的位置
                    touched_boundary = buffer_zone_current

                if touched_boundary:
                    # 2. 判断反弹（前一帧）球是否与边界相向运动
                    moving_towards_boundary = False
                    if touched_boundary == "left" and prev_vel_x < 0:
                        moving_towards_boundary = True
                    elif touched_boundary == "right" and prev_vel_x > 0:
                        moving_towards_boundary = True
                    elif touched_boundary == "top" and prev_vel_y < 0:
                        moving_towards_boundary = True
                    elif touched_boundary == "bottom" and prev_vel_y > 0:
                        moving_towards_boundary = True
                    # 3. 判断反弹后（当前帧）球是否与边界相离运动
                    moving_away_from_boundary = False
                    if touched_boundary == "left" and vel_x > 0:
                        moving_away_from_boundary = True
                    elif touched_boundary == "right" and vel_x < 0:
                        moving_away_from_boundary = True
                    elif touched_boundary == "top" and vel_y > 0:
                        moving_away_from_boundary = True
                    elif touched_boundary == "bottom" and vel_y < 0:
                        moving_away_from_boundary = True

                    # 如果满足前一帧与边界相向运动，当前帧与边界相离运动，继续检查速度分量变化是否合理
                    if moving_towards_boundary and moving_away_from_boundary:
                        # print(f"反弹监测：{frame_idx}帧-{ball_id}球,前一帧靠近,当前帧远离边界{touched_boundary}")

                        # 4. 检查速度分量变化是否合理
                        # 判断前一帧与当前帧速度垂直于边界的变化有没有相关性
                        if self.is_touched_boundary_and_vertical_velocity_reverse(touched_boundary, vel_x, vel_y, prev_vel_x, prev_vel_y):
                            print(f"---反弹监测：{frame_idx}帧-球{ball_id}在{touched_boundary}边界反弹,垂直分量反转")
                            # 如果满足前一帧与当前帧速度变化有相关性，记录反弹的球
                            rebounded_balls.append((ball_id,touched_boundary))
                        # 前一帧与当前帧速度变化没有相关性，有可能是当前帧垂直与边界的位移不明显，尝试比较平行于边界的速度分量是否一致
                        elif self.is_touched_boundary_and_parallel_velocity_same(touched_boundary, vel_x, vel_y, prev_vel_x, prev_vel_y):
                            print(f"---反弹监测：{frame_idx}帧-球{ball_id}在{touched_boundary}边界反弹,平行分量一致")
                            # 如果满足前一帧与当前帧速度变化有相关性，记录反弹的球
                            rebounded_balls.append((ball_id, touched_boundary))

                        else:
                            # 5.如果1和2成立但3和4不成立，检查是否位于袋口附近
                            is_near_the_hole_and_rebound, hole_name = self.is_near_the_hole_and_rebound(current_pos,previous_pos,velocity,previous_velocity,ball_id,frame_idx)
                            if is_near_the_hole_and_rebound:
                                print(f"---反弹监测：{frame_idx}帧-球{ball_id}在{hole_name}洞附近,碰撞库边反弹")
                                rebounded_balls.append((ball_id, touched_boundary))

        # 将这一帧中所有反弹的球的ID添加到self.ball_rebound中
        self.ball_rebound[frame_idx] = rebounded_balls

    def is_near_the_hole_and_rebound(self,current_pos,previous_pos,velocity,previous_velocity,ball_id,frame_idx):
        """
        检查球是否在洞口附近撞击桌边反弹
        1.在洞口附近
        2.前一帧与当前帧速度显著变化
        3.前一帧并不朝向任何球靠近
        4.当前帧不存在该球和其他球的碰撞
        """
        for hole_name, hole_position in self.hole_names_and_positions:
            is_near_hole, distance = self.is_near_hole(current_pos, hole_position)
            # 检查该球当前帧是否在洞口附近
            if is_near_hole:
                # 位于袋口附近，随即判断前后两帧速度是否显著变化
                velocity_change = self.get_velocity_change(velocity, previous_velocity)
                if velocity_change > self.ball_velocity_threshold:
                    # print(f"反弹监测：{frame_idx}帧-球{ball_id}在{hole_name}洞附近,可能碰撞库边反弹")

                    # 判断前一帧是否朝其他球运动
                    is_moving_towards = False
                    # 遍历所有球，检查前一帧是否有与当前球发生碰撞的可能
                    for other_ball_id, prev_other_position in self.balls_positions[frame_idx - 1].items():
                        if other_ball_id != ball_id and prev_other_position is not None:
                            distance = np.linalg.norm(np.array(previous_pos) - np.array(prev_other_position))
                            # 如果该球与当前球的距离小于设定的可能碰撞的距离阈值，则进行进一步速度向量的检查
                            if distance < self.ball_distance_threshold:
                                # 获取另一个球前一帧速度向量和位置坐标
                                prev_other_position = self.balls_positions[frame_idx - 1].get(
                                    other_ball_id)
                                prev_other_velocity = self.balls_velocities[frame_idx - 1].get(
                                    other_ball_id)
                                # 判断两个球是否相向运动
                                is_moving_towards = self.is_moving_towards(previous_velocity,prev_other_velocity,
                                                                           previous_pos,prev_other_position)
                    if is_moving_towards:
                        # 排除与球相向运动但是撞到库边反弹的情况，直接检索是否出现了与其他球的碰撞
                        ball_collision = self.ball_collision.get(frame_idx)  # [(ball_id1,ball_id2),(ball_id3,ball_id4)...]
                        if ball_id in [ball_id1 for ball_id1, ball_id2 in ball_collision]:  # 检查是否存在与其他球的碰撞
                            return False, None  # 在洞口附近撞击其他球，而非洞口附近库边反弹
                        else:
                            return True, hole_name # 没有在洞口附近撞击其他求，应该就是反弹
                    else:  # 在洞口附近没有撞击其他球
                        return True, hole_name  # 在洞口附近速度异常且前一帧没有朝其他球运动，应该就是反弹
                else:
                    return False, None # 速度没有发生异常，不属于在洞口附近反弹
            else:
                return False, None # 不在洞口附近，则不属于在洞口附近反弹

    def is_touched_boundary_and_parallel_velocity_same(self, boundary, vel_x, vel_y, prev_vel_x, prev_vel_y):
        """比较平行于边界的速度分量是否一致"""
        if boundary in ["left", "right"]:
            # 比较y方向速度是否保持一致
            return (abs(vel_y) > abs((1 - self.rebound_velocity_threshold) * prev_vel_y) and
                    abs(vel_y) < abs(1.1 * prev_vel_y))
        elif boundary in ["top", "bottom"]:
            # 比较x方向速度是否保持一致
            return (abs(vel_x) > abs((1 - self.rebound_velocity_threshold) * prev_vel_x) and
                    abs(vel_x) < abs(1.1 * prev_vel_x))
        return False

    def is_touched_boundary_and_vertical_velocity_reverse(self, touched_boundary, vel_x, vel_y, prev_vel_x, prev_vel_y):
        """比较垂直于边界的速度分量是否反向且基本相等"""
        if touched_boundary in ["left", "right"]:
            # 检查x方向速度反弹是否对称
            return (abs(prev_vel_x) > (1 - self.rebound_velocity_threshold) * abs(vel_x) and
                    abs(prev_vel_x) < (1 + self.rebound_velocity_threshold) * abs(vel_x))
        elif touched_boundary in ["top", "bottom"]:
            # 检查y方向速度反弹是否对称
            return (abs(prev_vel_y) > (1 - self.rebound_velocity_threshold) * abs(vel_y) and
                    abs(prev_vel_y) < (1 + self.rebound_velocity_threshold) * abs(vel_y))
        return False

    # 判断坐标是否在缓冲区域（在大框内但不在小框内）
    def is_in_buffer_zone(self, x, y):
        """判断坐标是否在缓冲区域内，返回最近的边界"""
        # 从self.effective_boundary获取基础撞击边界（桌面有效边界）
        left_buffer, right_buffer, top_buffer, bottom_buffer = self.effective_boundary
        # 根据 margin 调整真实边界
        left = left_buffer - self.margin
        right = right_buffer + self.margin
        top = top_buffer - self.margin
        bottom = bottom_buffer + self.margin
        # print(f"{x},{y}")
        # print(f"真实边界：{left, right, top, bottom}，缓冲边界：{left_buffer, right_buffer, top_buffer, bottom_buffer}")
        # (178.25, 1832.0, 111.0, 976.0),(278.25, 1732.0, 211.0, 876.0)

        # 判断坐标是否位于靠近桌边范围内
        if left < x < left_buffer or right_buffer < x < right or top < y < top_buffer or bottom_buffer < y < bottom:
            # print(f"坐标 ({x}, {y}) 在缓冲区域内")
            # 计算距离各个边界的距离
            distances = {
                "left": abs(x - left_buffer),
                "right": abs(x - right_buffer),
                "top": abs(y - top_buffer),
                "bottom": abs(y - bottom_buffer)
            }
            # 找到最近的边界
            closest_boundary = min(distances, key=distances.get)
            # print(f"最近的边界是: {closest_boundary}")
            return closest_boundary
        else:
            return None

    # （下）运行后处理 ----------------------------------------------------------------------------------------------------

    def run(self, segments_dict_pkl, time_interval=1.0):
        """对整个视频进行后处理条件判断"""
        # 加载分割结果字典
        video_segments = self.load_video_segments(segments_dict_pkl)

        for frame_idx, segments in sorted(video_segments.items()):
            # 计算当前帧中所有球的位置
            self.balls_positions[frame_idx] = self.process_frame_positions(frame_segments=segments)
            # print(f"{frame_idx}帧：所有球位置{self.balls_positions[frame_idx]}")
            # print(f"{frame_idx}帧：球9位置{self.balls_positions[frame_idx][9]}")

            if frame_idx > 0: # 从第二帧开始才有可能计算速度向量
                # 计算当前帧所有球的速度向量
                self.balls_velocities[frame_idx] = self.process_frame_velocities(frame_idx, time_interval)
                # print(f"{frame_idx}帧：所有球速度{self.balls_velocities[frame_idx]}")

                # 检查这一帧是否有球进洞
                self.check_ball_disappeared_pot(frame_idx)

                if frame_idx > 1:  # 要计算速度向量的变化差值需要比计算速度向量后一帧
                    # 检查这一帧是否有球间碰撞
                    self.check_ball_collision(frame_idx)
                    # 检查这一帧是否有球在桌边反弹
                    self.check_ball_rebound(frame_idx)



if __name__ == '__main__':
    post_processor = VideoPostProcessor()
    poket_pkl = "/root/autodl-tmp/temp_output/special_classes_detection.pkl"
    segments_dict_pkl = "/root/autodl-tmp/temp_output/video_segments.pkl"
    # 如果需要可视化 原始视频
    video_path = '/root/autodl-tmp/data/Det-SAM2评估集/videos/video149.mp4'
    # 可视化输出视频文件夹
    output_video_dir = "/root/autodl-tmp/temp_output"

    # 将袋口坐标分配給指定袋口
    post_processor.get_hole_name(poket_pkl)
    for hole_name, center in post_processor.hole_names_and_positions:
        print(f"---中心坐标: {center} 属于袋口: {hole_name}")
    # 根据袋口坐标计算桌面有效边界
    post_processor.get_boundary_from_holes()

    # 执行后处理条件判断主干函数
    post_processor.run(segments_dict_pkl)
    # 后处理可视化
    post_processor.visualize(video_source=video_path, output_video_dir=output_video_dir)
