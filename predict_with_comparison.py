import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import argparse
import cfg
from dataset import LoadDataset
from Models import  UNet, Deeplab_v3plus, SegFormer, HRNet, SDHRNet, SDUNet, SDDeepLabv3plus, AblationHRNet, SegNeXt

# 解析命令行参数
parser = argparse.ArgumentParser(description='Semantic Segmentation Prediction with Comparison View')
parser.add_argument('--model', type=str, default='UNet', choices=['UNet', 'Deeplab_v3plus', 'SegFormer', 'HRNet', 'SDHRNet', 'SDUNet', 'SDDeepLabv3plus', 'HRNetWithSemanticBranch', 'HRNetWithDetailBranch', 'HRNetWithSimpleFusion', 'UNetFormer', 'Mask2Former', 'ShapeAwareUNetFormer', 'SegNeXt'],
                    help='选择要使用的模型')
parser.add_argument('--weight_path', type=str, default=None, help='模型权重文件路径')
parser.add_argument('--num_samples', type=int, default=10, help='要预测的样本数量')
parser.add_argument('--all_samples', action='store_true', help='预测整个测试集')
parser.add_argument('--compare_mode', type=str, default='side_by_side', choices=['side_by_side', 'overlay', 'quad_view'],
                    help='对比显示模式: side_by_side(并排显示), overlay(叠加显示), quad_view(四视图显示)')
args = parser.parse_args()

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'使用设备: {device}')

# 加载测试数据
Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Load_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

# 根据命令行参数选择模型

if args.model == 'UNet':
    model = UNet.UNet(3, cfg.DATASET[1])
    model_name = 'UNet'
elif args.model == 'Deeplab_v3plus':
    model = Deeplab_v3plus.DeepLabv3_plus(nInputChannels=3, n_classes=cfg.DATASET[1], os=16, _print=False)
    model_name = 'Deeplab_v3plus'
elif args.model == 'SegFormer':
    # 根据配置选择SegFormer模型版本
    if cfg.SEGFORMER_VERSION == 'B0':
        model = SegFormer.SegFormerB0(num_classes=cfg.DATASET[1])
        model_name = 'SegFormerB0'
    elif cfg.SEGFORMER_VERSION == 'B1':
        model = SegFormer.SegFormerB1(num_classes=cfg.DATASET[1])
        model_name = 'SegFormerB1'
    elif cfg.SEGFORMER_VERSION == 'B2':
        model = SegFormer.SegFormerB2(num_classes=cfg.DATASET[1])
        model_name = 'SegFormerB2'
    elif cfg.SEGFORMER_VERSION == 'B3':
        model = SegFormer.SegFormerB3(num_classes=cfg.DATASET[1])
        model_name = 'SegFormerB3'
    elif cfg.SEGFORMER_VERSION == 'B4':
        model = SegFormer.SegFormerB4(num_classes=cfg.DATASET[1])
        model_name = 'SegFormerB4'
    elif cfg.SEGFORMER_VERSION == 'B5':
        model = SegFormer.SegFormerB5(num_classes=cfg.DATASET[1])
        model_name = 'SegFormerB5'
    else:
        # 默认使用B2版本
        model = SegFormer.SegFormerB2(num_classes=cfg.DATASET[1])
        model_name = 'SegFormerB2'
elif args.model == 'HRNet':
    model = HRNet.HRNet(num_classes=cfg.DATASET[1])
    model_name = 'HRNet'
elif args.model == 'SDHRNet':
    model = SDHRNet.SDHRNet(num_classes=cfg.DATASET[1])
    model_name = 'SDHRNet'
elif args.model == 'SDUNet':
    model = SDUNet.SDUNet(in_channel=3, out_channel=cfg.DATASET[1])
    model_name = 'SDUNet'
elif args.model == 'SDDeepLabv3plus':
    model = SDDeepLabv3plus.SDDeepLabv3plus(n_classes=cfg.DATASET[1])
    model_name = 'SDDeepLabv3plus'
elif args.model == 'HRNetWithSemanticBranch':
    model = AblationHRNet.HRNetWithSemanticBranch(num_classes=cfg.DATASET[1])
    model_name = 'HRNetWithSemanticBranch'
elif args.model == 'HRNetWithDetailBranch':
    model = AblationHRNet.HRNetWithDetailBranch(num_classes=cfg.DATASET[1])
    model_name = 'HRNetWithDetailBranch'
elif args.model == 'HRNetWithSimpleFusion':
    model = AblationHRNet.HRNetWithSimpleFusion(num_classes=cfg.DATASET[1])
    model_name = 'HRNetWithSimpleFusion'
elif args.model == 'SegNeXt':
    model = SegNeXt.SegNeXTB(num_classes=cfg.DATASET[1])
    model_name = 'SegNeXt'

# 动态设备选择与内存优化
try:
    model = model.to(device)
    if device.type == 'cuda':
        # 尝试使用半精度计算以减少内存使用
        model = model.half()
except RuntimeError as e:
    if 'out of memory' in str(e):
        print("CUDA内存不足，尝试使用CPU...")
        device = t.device('cpu')
        model = model.to(device)
    else:
        raise

# 加载模型权重
if args.weight_path:
    weight_path = args.weight_path
else:
    # 默认权重路径
    weight_dir = f'./Results/weights/{model_name}_weight/'
    if os.path.exists(weight_dir):
        # 获取最新的权重文件
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pth')]
        if weight_files:
            weight_files.sort(key=lambda x: int(x.split('.')[0]))
            weight_path = os.path.join(weight_dir, weight_files[-1])
        else:
            raise FileNotFoundError(f'在 {weight_dir} 中未找到权重文件')
    else:
        raise FileNotFoundError(f'权重目录 {weight_dir} 不存在')

# 加载权重时捕获内存错误
try:
    model.load_state_dict(t.load(weight_path, map_location=device))
except RuntimeError as e:
    if 'out of memory' in str(e):
        print("加载权重时内存不足，尝试清理内存并再次加载...")
        t.cuda.empty_cache() if device.type == 'cuda' else None
        import gc
        gc.collect()
        model.load_state_dict(t.load(weight_path, map_location='cpu'))
        device = t.device('cpu')
        model = model.to(device)
    else:
        raise

model.eval()

print(f'使用模型: {model_name}')
print(f'加载权重: {weight_path}')
print(f'当前数据集: {cfg.DATASET[0]}')
print(f'对比模式: {args.compare_mode}')

# 定义高对比度颜色映射
def get_high_contrast_colormap(dataset_name):
    # 为每个数据集定义高对比度颜色方案
    if dataset_name == 'Greenhouse':
        # Greenhouse数据集的高对比度颜色方案
        # 原始颜色: [0,0,0], [1,1,1], [2,2,2] (对比度非常低)
        # 新颜色: 黑色、鲜艳的绿色、鲜艳的蓝色（高对比度）
        return [
            [0, 0, 0],       # non_greenhouse (黑色)
            [0, 255, 0],     # plastic_greenhouse (亮绿色)
            [0, 0, 255]      # glass_greenhouse (亮蓝色)
        ]
    else:
        # 默认高对比度颜色方案
        num_classes = cfg.DATASET[1]
        colors = []
        # 生成一组高对比度的颜色
        for i in range(num_classes):
            # 使用HSV颜色空间生成均匀分布的颜色
            hue = i / num_classes
            saturation = 1.0
            value = 1.0
            
            # HSV转RGB
            c = value * saturation
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = value - c
            
            if 0 <= hue < 1/6:
                r, g, b = c, x, 0
            elif 1/6 <= hue < 2/6:
                r, g, b = x, c, 0
            elif 2/6 <= hue < 3/6:
                r, g, b = 0, c, x
            elif 3/6 <= hue < 4/6:
                r, g, b = 0, x, c
            elif 4/6 <= hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            colors.append([r, g, b])
        return colors

# 获取类别名称和高对比度颜色映射
def get_class_names_and_colormap(dataset_name, class_dict_path):
    # 读取类别名称
    class_names = []
    try:
        with open(class_dict_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                class_names.append(row[0])
    except Exception as e:
        print(f"读取类别名称时出错: {e}")
        # 如果无法读取，使用默认名称
        for i in range(cfg.DATASET[1]):
            class_names.append(f'Class_{i}')
    
    # 获取高对比度颜色映射
    high_contrast_colormap = get_high_contrast_colormap(dataset_name)
    
    # 确保颜色数量与类别数量匹配
    if len(high_contrast_colormap) != len(class_names):
        # 如果不匹配，使用默认的高对比度颜色生成
        num_classes = len(class_names)
        high_contrast_colormap = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 1.0
            value = 1.0
            
            # HSV转RGB
            c = value * saturation
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = value - c
            
            if 0 <= hue < 1/6:
                r, g, b = c, x, 0
            elif 1/6 <= hue < 2/6:
                r, g, b = x, c, 0
            elif 2/6 <= hue < 3/6:
                r, g, b = 0, c, x
            elif 3/6 <= hue < 4/6:
                r, g, b = 0, x, c
            elif 4/6 <= hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            high_contrast_colormap.append([r, g, b])
    
    return class_names, high_contrast_colormap

# 获取类别名称和高对比度颜色映射
class_names, high_contrast_colormap = get_class_names_and_colormap(cfg.DATASET[0], cfg.class_dict_path)
cm = np.array(high_contrast_colormap).astype('uint8')

# 创建保存结果的目录
result_dir = f"./Results/prediction_comparison/{model_name}_{args.compare_mode}/"
os.makedirs(result_dir, exist_ok=True)

# 打印颜色方案信息
print(f"\n当前使用的高对比度颜色方案 ({cfg.DATASET[0]} 数据集):")
for i, (name, color) in enumerate(zip(class_names, high_contrast_colormap)):
    print(f"{i}. {name}: RGB({color[0]}, {color[1]}, {color[2]})")

# 创建对比图像的函数
def create_comparison_image(original_img, true_label_img, pred_label_img, mode='side_by_side'):
    """创建不同模式的对比图像"""
    # 如果输入是numpy数组，转换为PIL图像
    if isinstance(original_img, np.ndarray):
        original_img = Image.fromarray(original_img)
    if isinstance(true_label_img, np.ndarray):
        true_label_img = Image.fromarray(true_label_img)
    if isinstance(pred_label_img, np.ndarray):
        pred_label_img = Image.fromarray(pred_label_img)
    
    width, height = original_img.size
    
    if mode == 'side_by_side':
        # 并排显示：原始图像 | 真实标签 | 预测结果
        comparison_img = Image.new('RGB', (width * 3, height), (255, 255, 255))
        comparison_img.paste(original_img, (0, 0))
        comparison_img.paste(true_label_img, (width, 0))
        comparison_img.paste(pred_label_img, (width * 2, 0))
        
        # 添加标签
        draw = ImageDraw.Draw(comparison_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            # 如果找不到Arial字体，使用默认字体
            font = ImageFont.load_default()
        
        draw.text((width // 2 - 30, 10), "原始图像", fill=(255, 255, 255), font=font)
        draw.text((width * 3 // 2 - 30, 10), "真实标签", fill=(255, 255, 255), font=font)
        draw.text((width * 5 // 2 - 30, 10), "预测结果", fill=(255, 255, 255), font=font)
        
    elif mode == 'overlay':
        # 叠加显示：预测结果与真实标签叠加
        comparison_img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        
        # 将真实标签和预测结果转换为RGBA
        true_label_rgba = true_label_img.convert('RGBA')
        pred_label_rgba = pred_label_img.convert('RGBA')
        
        # 设置透明度
        true_label_rgba.putalpha(128)  # 50%透明度
        pred_label_rgba.putalpha(128)  # 50%透明度
        
        # 叠加图像
        comparison_img.paste(original_img.convert('RGBA'), (0, 0))
        comparison_img.paste(true_label_rgba, (0, 0), true_label_rgba)
        comparison_img.paste(pred_label_rgba, (0, 0), pred_label_rgba)
        
        # 转换回RGB
        comparison_img = comparison_img.convert('RGB')
        
        # 添加标签
        draw = ImageDraw.Draw(comparison_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "叠加显示 (原始图像 + 真实标签 + 预测结果)", fill=(255, 255, 255), font=font)
        
    elif mode == 'quad_view':
        # 四视图显示：原始图像 | 真实标签 | 预测结果 | 差异图
        comparison_img = Image.new('RGB', (width * 2, height * 2), (255, 255, 255))
        comparison_img.paste(original_img, (0, 0))
        comparison_img.paste(true_label_img, (width, 0))
        comparison_img.paste(pred_label_img, (0, height))
        
        # 创建差异图
        true_label_np = np.array(true_label_img)
        pred_label_np = np.array(pred_label_img)
        
        # 计算差异（不匹配的像素设为红色）
        diff_mask = np.any(true_label_np != pred_label_np, axis=-1)
        diff_img = pred_label_np.copy()
        diff_img[diff_mask] = [255, 0, 0]  # 红色标记差异
        
        # 转换回PIL图像
        diff_img_pil = Image.fromarray(diff_img)
        comparison_img.paste(diff_img_pil, (width, height))
        
        # 添加标签
        draw = ImageDraw.Draw(comparison_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((width // 2 - 30, 10), "原始图像", fill=(255, 255, 255), font=font)
        draw.text((width * 3 // 2 - 30, 10), "真实标签", fill=(255, 255, 255), font=font)
        draw.text((width // 2 - 30, height + 10), "预测结果", fill=(255, 255, 255), font=font)
        draw.text((width * 3 // 2 - 30, height + 10), "差异图", fill=(255, 255, 255), font=font)
    
    return comparison_img

# 确定要预测的样本数量
if args.all_samples:
    num_to_predict = len(test_data)
else:
    num_to_predict = args.num_samples

# 进行预测
print(f"\n开始预测 {num_to_predict} 个样本...")

for i, sample in enumerate(test_data):
    if i >= num_to_predict:
        break
    
    try:
        valImg = sample['img'].to(device)
        valLabel = sample['label'].long().to(device)
        
        if device.type == 'cuda':
            valImg = valImg.half()
        
        # 内存清理
        t.cuda.empty_cache() if device.type == 'cuda' else None
        
        # 预测
        with t.no_grad():
            out = model(valImg)
            out = F.log_softmax(out, dim=1)
            pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        
        # 获取真实标签
        true_label = valLabel.squeeze().cpu().data.numpy()
        
        # 将原始图像转换回PIL格式
        original_img = valImg.squeeze().permute(1, 2, 0).cpu().data.numpy()
        original_img = (original_img * 255).astype('uint8')
        original_img = Image.fromarray(original_img)
        
        # 应用高对比度颜色映射到预测结果和真实标签
        pred_colored = cm[pre_label]
        true_colored = cm[true_label]
        
        # 创建对比图像
        comparison_img = create_comparison_image(original_img, true_colored, pred_colored, args.compare_mode)
        
        # 保存对比图像
        save_path = os.path.join(result_dir, f"{i}_comparison.png")
        comparison_img.save(save_path)
        
        # 也保存单独的预测和标签图像以便需要时查看
        pred_img = Image.fromarray(pred_colored)
        true_img = Image.fromarray(true_colored)
        pred_img.save(os.path.join(result_dir, f"{i}_prediction.png"))
        true_img.save(os.path.join(result_dir, f"{i}_label.png"))
        
        print(f'已保存对比结果 {i+1}/{num_to_predict}: {save_path}')
        
        # 清理变量以释放内存
        del valImg, valLabel, out, pre_label, true_label, original_img, pred_colored, true_colored, comparison_img
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"处理样本 {i} 时出错: {e}")
        # 继续处理下一个样本
        continue

print(f"\n预测完成！所有结果已保存到 {result_dir}")
print(f"使用的对比模式: {args.compare_mode}")
print("提示：对比图像使用了高对比度颜色方案，使不同类别的区域更加容易区分，便于直观评估模型性能。")