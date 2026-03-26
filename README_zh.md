# 语义-细节双流解耦 HRNet 项目

## 项目简介

本项目实现了一种语义-细节双流解耦的HRNet模型（SDHRNet），用于温室的语义分割任务。该模型通过分离语义特征和细节特征的提取过程，提高了对不同类型温室的分割精度。

## 核心文件说明

### 1. 数据处理与分析

#### select_samples.py
- **功能**：从数据集中选择高质量的样本用于频率分析
- **作用**：筛选出包含塑料拱棚和日光温室的样本，并确保样本中目标类别占比足够大
- **参数**：
  - `dataset_root`：数据集根路径
  - `output_root`：输出目录
  - `target_count`：目标样本数量
  - `area_threshold`：面积阈值，确保目标区域足够大
- **输出**：
  - `selected_samples_for_fft/plastic_shed`：塑料拱棚样本
  - `selected_samples_for_fft/solar_greenhouse`：日光温室样本

#### analyze_frequency.py
- **功能**：分析两种温室类型的频率特征
- **作用**：通过FFT和拉普拉斯方差计算，分析塑料拱棚（高频特征）和日光温室（低频特征）的频率特性
- **参数**：
  - `data_root`：输入数据路径
  - `output_dir`：输出目录
  - `PATCH_SIZE`：切片大小
- **输出**：
  - 塑料拱棚和日光温室的频谱图
  - 频率差异热力图
  - 径向分布曲线图
  - 拉普拉斯方差统计结果

### 2. 模型配置与训练

#### cfg.py
- **功能**：项目基础配置
- **配置项**：
  - `BATCH_SIZE`：批量大小
  - `EPOCH_NUMBER`：训练轮数
  - `DATASET`：数据集配置（名称和类别数）
  - `SEGFORMER_VERSION`：SegFormer模型版本
  - `crop_size`：裁剪尺寸
  - 数据集路径：训练、验证、测试集的图像和标签路径

#### train.py
- **功能**：模型训练
- **支持的模型**：UNet、Deeplab_v3plus、SegFormer、HRNet、SDHRNet等
- **训练过程**：
  - 早停机制
  - 学习率调整
  - 梯度裁剪
  - 训练和验证指标计算
- **输出**：
  - 模型权重文件：`./Results/weights/{model_name}_weight/`
  - 训练日志和指标

### 3. 模型测试与评估

#### test.py
- **功能**：模型测试和评估
- **评估指标**：
  - 交并比（IoU）
  - 像素精度
  - 类别精度
  - 精确率、召回率、F1分数
- **输出**：
  - 详细评估指标：`./Results/metrics/{model_name}_metrics.txt`
  - 混淆矩阵：`./Results/confusion_matrix/{model_name}_confusion_matrix.png`

#### predict_with_comparison.py
- **功能**：模型预测和结果对比
- **对比模式**：
  - `side_by_side`：并排显示原始图像、真实标签和预测结果
  - `overlay`：叠加显示
  - `quad_view`：四视图显示（原始图像、真实标签、预测结果、差异图）
- **输出**：
  - 对比图像：`./Results/prediction_comparison/{model_name}_{compare_mode}/`
  - 单独的预测和标签图像

#### visualize_feature_maps.py
- **功能**：生成模型特征图热力图
- **可视化内容**：
  - 原始图像
  - 真实标签
  - 预测结果
  - 语义分支特征图
  - 细节分支特征图
  - 互补机制展示图
- **输出**：
  - 热力图：`./Results/feature_maps_jet/`

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- tqdm

## 使用指南

### 1. 数据准备

1. 按照 `cfg.py` 中的路径配置数据集
2. 运行 `select_samples.py` 选择用于频率分析的样本：
   ```bash
   python select_samples.py
   ```

### 2. 频率分析

运行 `analyze_frequency.py` 分析两种温室的频率特征：
```bash
python analyze_frequency.py
```

### 3. 模型训练

运行 `train.py` 训练模型，指定模型类型：
```bash
python train.py --model SDHRNet --patience 30 --monitor miou
```

### 4. 模型测试

运行 `test.py` 测试模型性能：
```bash
python test.py --model SDHRNet
```

使用指定权重文件进行测试：
```bash
python test.py --model SDHRNet --weight_path Results\weights\SDHRNet_weight\52.pth
```

### 5. 预测与可视化

运行 `predict_with_comparison.py` 生成预测结果对比：
```bash
python predict_with_comparison.py --model SDHRNet --compare_mode quad_view --num_samples 5
```

使用指定权重文件进行预测：
```bash
python predict_with_comparison.py --model SDHRNet --weight_path Results\weights\SDHRNet_weight\52.pth --compare_mode quad_view --num_samples 5
```

运行 `visualize_feature_maps.py` 生成特征图热力图：
```bash
python visualize_feature_maps.py
```

## 结果存放

| 结果类型 | 存放路径 | 说明 |
|---------|---------|------|
| 模型权重 | `./Results/weights/{model_name}_weight/` | 训练过程中保存的模型权重 |
| 评估指标 | `./Results/metrics/{model_name}_metrics.txt` | 详细的模型评估指标 |
| 混淆矩阵 | `./Results/confusion_matrix/{model_name}_confusion_matrix.png` | 模型预测的混淆矩阵 |
| 预测对比 | `./Results/prediction_comparison/{model_name}_{compare_mode}/` | 不同模式的预测结果对比 |
| 特征图热力图 | `./Results/feature_maps_jet/` | 语义分支和细节分支的特征图热力图 |
| 频率分析结果 | `./frequency_analysis_final/` | 温室频率特征分析结果 |

## 模型架构

SDHRNet 模型由以下部分组成：
1. HRNet W48 主干网络
2. 多尺度语义分支：使用不同大小的卷积核捕捉语义特征
3. 空洞细节分支：使用空洞卷积和池化操作捕捉细节特征
4. 自适应融合模块：根据特征重要性动态调整权重
5. 分类层：生成最终的分割结果

## 消融实验

为了评估 SDHRNet 中各组件的贡献，项目包含了在 `Models/AblationHRNet.py` 中实现的消融实验：

### 1. HRNetWithSemanticBranch
- **描述**：仅包含多尺度语义分支的 HRNet 骨干网络
- **目的**：评估语义分支单独的有效性
- **命令示例**：
  ```bash
  python train.py --model HRNetWithSemanticBranch
  python test.py --model HRNetWithSemanticBranch
  ```

### 2. HRNetWithDetailBranch
- **描述**：仅包含空洞细节分支的 HRNet 骨干网络
- **目的**：评估细节分支单独的有效性
- **命令示例**：
  ```bash
  python train.py --model HRNetWithDetailBranch
  python test.py --model HRNetWithDetailBranch
  ```

### 3. HRNetWithSimpleFusion
- **描述**：同时包含语义和细节分支的 HRNet 骨干网络，但使用简单的加法融合而不是自适应融合
- **目的**：评估自适应融合机制的影响
- **命令示例**：
  ```bash
  python train.py --model HRNetWithSimpleFusion
  python test.py --model HRNetWithSimpleFusion
  ```

## 技术特点

1. **双流解耦**：分离语义特征和细节特征的提取，分别优化
2. **多尺度特征**：利用HRNet的多尺度特征表示能力
3. **自适应融合**：根据不同区域的特征重要性动态调整融合权重
4. **频率感知**：基于频率分析结果设计网络结构，针对不同类型温室的特征特点进行优化

## 注意事项

1. 确保数据集按照指定格式组织
2. 训练前可根据硬件条件调整 `BATCH_SIZE`
3. 测试时如遇内存不足，可尝试使用 `--batch_size 1`
4. 可视化生成的热力图和对比图像可直接用于论文或报告

## 许可证

本项目仅供学术研究使用。