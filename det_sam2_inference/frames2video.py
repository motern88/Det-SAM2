import cv2
import os
import subprocess

def frames_to_video(frames_folder, output_video_path, fps=30):
    # 获取帧文件列表并按名称排序
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith(".png")]
    if not frame_files:
        raise ValueError(f"No frames found in {frames_folder}. Please check the folder.")
    frame_files.sort()  # 确保按顺序排序，如 frame_00000.png, frame_00001.png

    # 读取第一帧以确定视频的宽高
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape

    # 初始化视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用  编码
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧写入视频文件
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)  # 将帧写入视频

    # 释放视频文件
    video.release()
    print(f"将{frames_folder}路径帧合成视频，已保存为 {output_video_path}")

if __name__ == '__main__':
    # 使用函数将帧合成视频
    frames_folder = '/root/autodl-tmp/prompt_results' # 'det-sam2_demo_output'  # 视频帧所在文件夹
    output_video_path = '/root/autodl-tmp/prompt_visual.mp4' # 'videos/output_video.mp4'  # 输出视频文件路径

    # 合成视频
    frames_to_video(frames_folder, output_video_path, fps=2)  # fps=60

