import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
np.random.seed(3)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# 设置文件夹路径
images_folder = '/root/数据集修正/images'
labels_folder = '/root/数据集修正/labels'
output_folder = '/root/数据集修正/output_labels'
visual_folder = '/root/数据集修正/visualization'
os.makedirs(output_folder, exist_ok=True)

print(f"读取图像路径：{images_folder}"
      f"读取标注路径：{labels_folder}")

# 转换归一化坐标为原始坐标的函数
def convert_normalized_to_original_bbox(normalized_bbox, image_width, image_height):
    class_id, cx, cy, w, h = normalized_bbox
    original_cx = cx * image_width
    original_cy = cy * image_height
    original_w = w * image_width
    original_h = h * image_height
    x1 = int(original_cx - original_w / 2)
    y1 = int(original_cy - original_h / 2)
    x2 = int(original_cx + original_w / 2)
    y2 = int(original_cy + original_h / 2)
    return (class_id, x1, y1, x2, y2)

# 获取修正的边界框
def get_corrected_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return (x, y, x + w, y + h)

# 可视化标注框的函数
def visualize_bounding_boxes(image, original_boxes, corrected_boxes, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # 绘制所有原始标注框（黄）
    for box in original_boxes:
        plt.gca().add_patch(plt.Rectangle((box[1], box[2]),
                                            box[3] - box[1],
                                            box[4] - box[2],
                                            edgecolor='yellow', facecolor='none', linewidth=0.5))

    # 绘制所有修正标注框（红色）
    for box in corrected_boxes:
        plt.gca().add_patch(plt.Rectangle((box[1], box[2]),
                                            box[3] - box[1],
                                            box[4] - box[2],
                                            edgecolor='red', facecolor='none', linewidth=0.5))

    plt.axis('off')
    plt.title('Original and Corrected Bounding Boxes')
    plt.savefig(output_path)  # 保存到文件
    plt.close()  # 关闭图形

# SAM2配置
sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# 处理每个图像和其对应的标签
for label_file in tqdm(os.listdir(labels_folder), desc="Processing labels"):
    if label_file.endswith('.txt'):
        image_name = label_file.replace('.txt', '.jpg')  # 假设图片为.jpg格式
        image_path = os.path.join(images_folder, image_name)
        label_file_path = os.path.join(labels_folder, label_file)

        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            continue

        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        image_height, image_width, _ = image.shape

        # SAM2设置图像
        predictor.set_image(image)

        original_bboxes = []  # 存储原始边界框
        corrected_bboxes = []

        # 读取标注并处理
        with open(label_file_path, 'r') as file:
            for line in file:
                if line.strip():
                    bounding_box = list(map(float, line.split()))
                    if len(bounding_box) != 5:  # 跳过不为5个数的行
                        continue
                    input_box = convert_normalized_to_original_bbox(bounding_box, image_width, image_height)
                    # print(input_box)
                    original_bboxes.append(input_box)  # 保存原始边界框
                    input_box = np.array(input_box)

                    # SAM2 生成掩码
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[1:5][None, :],  # [1:5]只取坐标部分
                        multimask_output=False,
                    )

                    mask = masks[0]  # 获取第一个掩膜
                    mask_uint8 = (mask * 255).astype(np.uint8)  # 转换为 uint8 格式
                    corrected_box = get_corrected_bounding_box(mask_uint8)
                    if corrected_box is not None:
                        corrected_bboxes.append((input_box[0], *corrected_box))  # 添加类别和修正框

        # 写入新的标注文件
        output_label_file_path = os.path.join(output_folder, label_file)
        with open(output_label_file_path, 'w') as output_file:
            for class_id, x1, y1, x2, y2 in corrected_bboxes:
                output_file.write(f"{class_id} {x1} {y1} {x2} {y2}\n")

        # 在处理完所有框后可视化
        output_image_path = os.path.join(visual_folder, image_name.replace('.jpg', '_visualization.png'))
        # 确保输出文件夹存在
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        visualize_bounding_boxes(image, original_bboxes, corrected_bboxes, output_image_path)

print(f"修正标注结果保存路径：{output_folder}")
print(f"修正过程可视化保存路径：{visual_folder}")
