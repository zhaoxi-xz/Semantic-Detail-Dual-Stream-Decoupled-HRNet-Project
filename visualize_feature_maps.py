import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import cfg
from dataset import LoadDataset
from Models.SDHRNet import SDHRNet, adaptive_fusion

# 解析命令行参数
parser = argparse.ArgumentParser(description='Decoupled Feature Visualization (Jet Style)')
parser.add_argument('--model_path', type=str, default=None, help='模型权重路径')
parser.add_argument('--image_index', type=int, default=None, help='指定可视化的图像索引')
parser.add_argument('--save_dir', type=str, default='./Results/feature_maps_jet', help='保存目录')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'使用设备: {device}')
os.makedirs(args.save_dir, exist_ok=True)

# 1. 加载数据
Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Load_test, batch_size=1, shuffle=False, num_workers=0)

# 2. 加载模型
model = SDHRNet(num_classes=cfg.DATASET[1])
model = model.to(device)

if args.model_path:
    weight_path = args.model_path
else:
    # 自动寻找最新权重
    weight_dir = './Results/weights/ShapeAwareHRNet_weight/'
    if os.path.exists(weight_dir):
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pth')]
        if weight_files:
            weight_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
            weight_path = os.path.join(weight_dir, weight_files[-1])
        else:
            raise FileNotFoundError("未找到权重文件")
    else:
        raise FileNotFoundError("权重目录不存在")

print(f'加载权重: {weight_path}')
model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
model.eval()


# 3. 包装模型 (保持新的语义-细节互补逻辑)
class ModelWithFeatureReturn(nn.Module):
    def __init__(self, original_model):
        super(ModelWithFeatureReturn, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # Backbone
        multi_scale_features = self.original_model.backbone(x)
        high_res_feature = multi_scale_features[0]

        # Branches
        # 语义分支 (捕捉主体/日光温室)
        semantic_feat = self.original_model.semantic_branch(high_res_feature)

        # 细节分支 (捕捉纹理/塑料拱棚 + 边缘)
        detail_feat = self.original_model.detail_branch(high_res_feature)

        # Fusion
        shape_aware_feat = adaptive_fusion(semantic_feat, detail_feat)

        # Classifier
        output = self.original_model.classifier(shape_aware_feat)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

        return output, semantic_feat, detail_feat


model_with_features = ModelWithFeatureReturn(model)


# 4. 可视化处理函数 (修改了绘图配色和保存方式)
def process_and_visualize(index, sample):
    valImg = sample['img'].to(device)
    valLabel = sample['label'].long().to(device)

    with torch.no_grad():
        output, semantic_feat, detail_feat = model_with_features(valImg)

    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_img = valImg.squeeze().permute(1, 2, 0).cpu().numpy()
    original_img = std * original_img + mean
    original_img = np.clip(original_img, 0, 1)

    true_label = valLabel.squeeze().cpu().numpy()
    pred_label = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 自定义颜色映射：类别0=红色，类别1=绿色，类别2=蓝色
    def apply_custom_colormap(label_map):
        # 创建颜色映射
        colormap = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
        # 类别0: 红色
        colormap[label_map == 0] = [255, 0, 0]
        # 类别1: 绿色
        colormap[label_map == 1] = [0, 255, 0]
        # 类别2: 蓝色
        colormap[label_map == 2] = [0, 0, 255]
        return colormap

    # 热力图生成工具
    def get_heatmap(feat):
        heatmap = torch.mean(feat, dim=1).squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = Image.fromarray((heatmap * 255).astype('uint8'))
        heatmap = heatmap.resize((valImg.size(3), valImg.size(2)), Image.BILINEAR)
        return np.array(heatmap) / 255.0

    semantic_map = get_heatmap(semantic_feat)
    detail_map = get_heatmap(detail_feat)

    # 应用自定义颜色映射
    true_label_colored = apply_custom_colormap(true_label)
    pred_label_colored = apply_custom_colormap(pred_label)

    # === 分别保存六张图 ===
    # 1. 保存原始图像
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.axis('off')
    save_path = os.path.join(args.save_dir, f'original_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存原始图像: {save_path}')

    # 2. 保存真实标签图
    plt.figure(figsize=(6, 6))
    plt.imshow(true_label_colored)
    plt.axis('off')
    save_path = os.path.join(args.save_dir, f'ground_truth_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存真实标签图: {save_path}')

    # 3. 保存预测结果图
    plt.figure(figsize=(6, 6))
    plt.imshow(pred_label_colored)
    plt.axis('off')
    save_path = os.path.join(args.save_dir, f'prediction_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存预测结果图: {save_path}')

    # 4. 保存语义分支特征图
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img, alpha=1.0)
    im = plt.imshow(semantic_map, cmap='jet', alpha=0.5)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    save_path = os.path.join(args.save_dir, f'semantic_branch_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存语义分支特征图: {save_path}')

    # 5. 保存细节分支特征图
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img, alpha=1.0)
    im = plt.imshow(detail_map, cmap='jet', alpha=0.5)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    save_path = os.path.join(args.save_dir, f'detail_branch_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存细节分支特征图: {save_path}')

    # 6. 保存互补机制展示图
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img, alpha=0.6)
    plt.imshow(semantic_map, cmap='Reds', alpha=0.5)
    plt.imshow(detail_map, cmap='Blues', alpha=0.5)
    plt.axis('off')
    save_path = os.path.join(args.save_dir, f'complementarity_map_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存互补机制展示图: {save_path}')


# 5. 智能搜索
target_classes = [1, 2]
found_count = 0
max_samples = 3

print(f"正在搜索图像...")

for i, sample in enumerate(test_data):
    if args.image_index is not None:
        if i == args.image_index:
            process_and_visualize(i, sample)
            break
        continue

    label = sample['label']
    unique_classes = torch.unique(label)

    has_plastic = 1 in unique_classes
    has_glass = 2 in unique_classes

    if has_plastic or has_glass:
        print(f"Index {i}: 生成可视化...")
        process_and_visualize(i, sample)
        found_count += 1

    if found_count >= max_samples:
        print("结束。")
        break