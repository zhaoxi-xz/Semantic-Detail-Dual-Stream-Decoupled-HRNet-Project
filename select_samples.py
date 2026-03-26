import os
import cv2
import shutil
import random
from tqdm import tqdm
import numpy as np

# 输入路径定义
dataset_root = r"e:\desk\Semantic_Segmentation_FCN-master 2\Datasets\Greenhouse3"

# 输出路径定义
output_root = r"e:\desk\Semantic_Segmentation_FCN-master 2\selected_samples_for_fft"
plastic_output = os.path.join(output_root, "plastic_shed")
solar_output = os.path.join(output_root, "solar_greenhouse")

# 目标数量
target_count = 50

# 面积阈值：512x512x0.1
area_threshold = 512 * 512 * 0.1

# 数据集分割（train, val, test）
dataset_splits = ["train", "val", "test"]

# 创建输出目录结构
def create_output_dirs():
    os.makedirs(os.path.join(plastic_output, "images"), exist_ok=True)
    os.makedirs(os.path.join(plastic_output, "masks"), exist_ok=True)
    os.makedirs(os.path.join(solar_output, "images"), exist_ok=True)
    os.makedirs(os.path.join(solar_output, "masks"), exist_ok=True)

# 获取所有mask文件路径（包括train, val, test子目录）
def get_mask_files():
    mask_files = []
    for split in dataset_splits:
        mask_dir = os.path.join(dataset_root, split, "masks")
        if os.path.exists(mask_dir):
            # 只收集PNG格式的mask文件
            files = [os.path.join(split, "masks", f) for f in os.listdir(mask_dir) if f.endswith(".png")]
            mask_files.extend(files)
    return mask_files

# 计算三通道mask中各类别的像素数量
def count_classes(mask):
    # 背景：RGB(0,0,0)
    background = np.all(mask == [0, 0, 0], axis=-1)
    # 塑料大棚：RGB(1,1,1)
    plastic = np.all(mask == [1, 1, 1], axis=-1)
    # 日光温室：RGB(2,2,2)
    solar = np.all(mask == [2, 2, 2], axis=-1)
    
    counts = {
        0: np.sum(background),
        1: np.sum(plastic),
        2: np.sum(solar)
    }
    return counts

# 检查mask是否符合塑料拱棚条件
def is_good_plastic_shed(mask_path):
    # 读取三通道图像
    mask = cv2.imread(mask_path)
    if mask is None:
        return False
    
    # 计算各类别像素数量
    counts = count_classes(mask)
    
    # 检查面积阈值
    if counts[1] <= area_threshold:
        return False
    
    # 纯度检查：类别1至少是类别2的2倍，或类别2不存在
    if counts[2] == 0:
        return True
    return counts[1] >= counts[2] * 2

# 检查mask是否符合日光温室条件
def is_good_solar_greenhouse(mask_path):
    # 读取三通道图像
    mask = cv2.imread(mask_path)
    if mask is None:
        return False
    
    # 计算各类别像素数量
    counts = count_classes(mask)
    
    # 检查面积阈值
    if counts[2] <= area_threshold:
        return False
    
    # 纯度检查：类别2至少是类别1的2倍，或类别1不存在
    if counts[1] == 0:
        return True
    return counts[2] >= counts[1] * 2

# 复制文件
def copy_files(mask_file_path, is_plastic):
    # 构建源文件路径
    # mask_file_path格式：split/masks/filename.png
    # 提取split和filename
    parts = mask_file_path.split(os.sep)
    split = parts[0]  # train/val/test
    mask_filename = parts[-1]  # filename.png
    
    # 将PNG mask文件名转换为TIF图像文件名
    image_filename = mask_filename.replace(".png", ".tif")
    
    # 构建源图像和mask路径 - 图像文件直接在images目录下
    source_image = os.path.join(dataset_root, split, "images", image_filename)
    source_mask = os.path.join(dataset_root, mask_file_path)
    
    # 检查文件是否存在
    if not os.path.exists(source_image):
        print(f"Warning: Image file not found: {source_image}")
        return
    
    if not os.path.exists(source_mask):
        print(f"Warning: Mask file not found: {source_mask}")
        return
    
    # 构建目标路径
    if is_plastic:
        dest_image = os.path.join(plastic_output, "images", image_filename)
        dest_mask = os.path.join(plastic_output, "masks", mask_filename)
    else:
        dest_image = os.path.join(solar_output, "images", image_filename)
        dest_mask = os.path.join(solar_output, "masks", mask_filename)
    
    # 复制文件
    shutil.copy(source_image, dest_image)
    shutil.copy(source_mask, dest_mask)

# 主函数
def main():
    # 创建输出目录
    create_output_dirs()
    
    # 获取所有mask文件并打乱
    mask_files = get_mask_files()
    print(f"Total mask files found: {len(mask_files)}")
    random.shuffle(mask_files)
    
    # 统计变量
    plastic_count = 0
    solar_count = 0
    
    # 遍历所有mask文件
    for mask_file_path in tqdm(mask_files, desc="Processing masks"):
        if plastic_count >= target_count and solar_count >= target_count:
            break
        
        # 构建完整的mask路径
        full_mask_path = os.path.join(dataset_root, mask_file_path)
        
        # 检查是否是好的塑料拱棚样本
        if plastic_count < target_count and is_good_plastic_shed(full_mask_path):
            copy_files(mask_file_path, True)
            plastic_count += 1
        
        # 检查是否是好的日光温室样本
        if solar_count < target_count and is_good_solar_greenhouse(full_mask_path):
            copy_files(mask_file_path, False)
            solar_count += 1
    
    # 打印结果
    print(f"Found {plastic_count} plastic sheds, {solar_count} solar greenhouses")
    print(f"Results saved to {output_root}")

if __name__ == "__main__":
    main()