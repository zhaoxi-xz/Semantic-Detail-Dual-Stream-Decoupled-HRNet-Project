import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from evalution_segmentaion import eval_semantic_segmentation, calc_semantic_segmentation_confusion, calc_semantic_segmentation_iou
from dataset import LoadDataset
from Models import UNet, SDDeepLabv3plus, SDUNet, Deeplab_v3plus, SegFormer, HRNet, SDHRNet, SegNeXt, AblationHRNet
import cfg

# 主程序
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Semantic Segmentation Testing with IoU Fix')
    parser.add_argument('--model', type=str, default='UNet', choices=['SDUNet','SDDeepLabv3plus' , 'UNet', 'Deeplab_v3plus', 'SDHRNet', 'SegFormer', 'HRNet',  'HRNetWithSemanticBranch', 'HRNetWithDetailBranch', 'HRNetWithSimpleFusion', 'UNetFormer', 'ShapeAwareUNetFormer','Mask2Former',  'SegNeXt'],
                        help='选择要使用的模型')
    parser.add_argument('--weight_path', type=str, default=None, help='模型权重文件路径')
    args = parser.parse_args()

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    num_class = cfg.DATASET[1]

    BATCH_SIZE = 1
    miou_list = [0]

    print(f'开始加载测试数据集...')
    Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
    # 降低worker数量并禁用pin_memory以解决Windows上的多进程问题
    test_data = DataLoader(Load_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    print(f'测试数据集加载完成，共有{len(Load_test)}个样本')

    # 根据命令行参数选择模型
    
    if args.model == 'UNet':
        model = UNet.UNet(3, num_class)
        model_name = 'UNet'
    elif args.model == 'Deeplab_v3plus':
        model = Deeplab_v3plus.DeepLabv3_plus(nInputChannels=3, n_classes=num_class, os=16, _print=False)
        model_name = 'Deeplab_v3plus'
    elif args.model == 'SegFormer':
        # 根据配置选择SegFormer模型版本
        if cfg.SEGFORMER_VERSION == 'B0':
            model = SegFormer.SegFormerB0(num_classes=num_class)
            model_name = 'SegFormerB0'
        elif cfg.SEGFORMER_VERSION == 'B1':
            model = SegFormer.SegFormerB1(num_classes=num_class)
            model_name = 'SegFormerB1'
        elif cfg.SEGFORMER_VERSION == 'B2':
            model = SegFormer.SegFormerB2(num_classes=num_class)
            model_name = 'SegFormerB2'
        elif cfg.SEGFORMER_VERSION == 'B3':
            model = SegFormer.SegFormerB3(num_classes=num_class)
            model_name = 'SegFormerB3'
        elif cfg.SEGFORMER_VERSION == 'B4':
            model = SegFormer.SegFormerB4(num_classes=num_class)
            model_name = 'SegFormerB4'
        elif cfg.SEGFORMER_VERSION == 'B5':
            model = SegFormer.SegFormerB5(num_classes=num_class)
            model_name = 'SegFormerB5'
        else:
            # 默认使用B2版本
            model = SegFormer.SegFormerB2(num_classes=num_class)
            model_name = 'SegFormerB2'
    elif args.model == 'HRNet':
        model = HRNet.HRNet(num_classes=num_class)
        model_name = 'HRNet'
    elif args.model == 'HRNetWithSemanticBranch':
        model = AblationHRNet.HRNetWithSemanticBranch(num_classes=num_class)
        model_name = 'HRNetWithSemanticBranch'
    elif args.model == 'HRNetWithDetailBranch':
        model = AblationHRNet.HRNetWithDetailBranch(num_classes=num_class)
        model_name = 'HRNetWithDetailBranch'
    elif args.model == 'HRNetWithSimpleFusion':
        model = AblationHRNet.HRNetWithSimpleFusion(num_classes=num_class)
        model_name = 'HRNetWithSimpleFusion'
    elif args.model == 'SDHRNet':
        model = SDHRNet.SDHRNet(num_classes=num_class)
        model_name = 'SDHRNet'
    elif args.model == 'SDUNet':
        model = SDUNet.SDUNet(in_channel=3, out_channel=num_class)
        model_name = 'SDUNet'
    elif args.model == 'SDDeepLabv3plus':
        model = SDDeepLabv3plus.SDDeepLabv3plus(n_classes=num_class)
        model_name = 'SDDeepLabv3plus'
    elif args.model == 'SegNeXt':
        model = SegNeXt.SegNeXTB(num_classes=num_class)
        model_name = 'SegNeXt'
    
    model = model.to(device)
    model.eval()

    # 统一确定并加载模型权重（始终执行）
    weight_dir = os.path.join('Results', 'weights', f'{model_name}_weight')
    if args.weight_path:
        weight_path = args.weight_path
    else:
        if not os.path.exists(weight_dir):
            raise FileNotFoundError(f'权重目录 {weight_dir} 不存在')
        weights = sorted([os.path.join(weight_dir, f) for f in os.listdir(weight_dir) if f.endswith('.pth')])
        if not weights:
            raise FileNotFoundError(f'权重目录 {weight_dir} 下没有 .pth 文件')
        weight_path = weights[-1]

    try:
        ckpt = t.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = t.load(weight_path, map_location=device)

    # 兼容整模型保存或包含state_dict字段
    if not isinstance(ckpt, dict):
        try:
            ckpt = ckpt.state_dict()
        except Exception:
            ckpt = ckpt.get('state_dict', ckpt)

    # 处理DataParallel的'module.'前缀
    if isinstance(ckpt, dict):
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            nk = k[7:] if k.startswith('module.') else k
            new_ckpt[nk] = v
        ckpt = new_ckpt

    # ViT位置编码过滤与加载
    if args.model == 'FCN_ViT':
        if isinstance(ckpt, dict):
            ckpt.pop('encoder.pos_embedding', None)
        load_result = model.load_state_dict(ckpt, strict=False)
    else:
        try:
            load_result = model.load_state_dict(ckpt)  # strict=True 默认
        except RuntimeError as e:
            # 严格加载失败，降级为strict=False以继续测试并打印反馈
            print(f'严格加载失败: {str(e)[:200]}')
            load_result = model.load_state_dict(ckpt, strict=False)

    print(f'使用模型: {model_name}')
    print(f'加载权重: {weight_path}')
    # 显式反馈加载结果
    missing = getattr(load_result, 'missing_keys', [])
    unexpected = getattr(load_result, 'unexpected_keys', [])
    print(f'权重加载反馈: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}')
    if len(missing) > 0:
        print(f'missing_keys样例: {missing[:5]}')
    if len(unexpected) > 0:
        print(f'unexpected_keys样例: {unexpected[:5]}')
    net = model

    test_acc = 0
    test_miou = 0
    test_mpa = 0
    error = 0
    
    # 初始化混淆矩阵
    total_confusion = None
    all_preds = []
    all_labels = []
    all_pixel_accuracies = []
    all_class_accuracies = []

    for i, sample in enumerate(test_data):
        data = Variable(sample['img']).to(device)
        label = Variable(sample['label']).to(device)
        out = net(data)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = label.data.cpu().numpy()
        true_label = [i for i in true_label]

        # 计算当前批次的混淆矩阵并累加到总混淆矩阵
        current_confusion = calc_semantic_segmentation_confusion(pre_label, true_label)
        if total_confusion is None:
            total_confusion = current_confusion
        else:
            # 确保两个混淆矩阵大小相同
            if total_confusion.shape != current_confusion.shape:
                max_size = max(total_confusion.shape[0], current_confusion.shape[0])
                new_total = np.zeros((max_size, max_size), dtype=np.int64)
                new_total[:total_confusion.shape[0], :total_confusion.shape[1]] = total_confusion
                new_current = np.zeros((max_size, max_size), dtype=np.int64)
                new_current[:current_confusion.shape[0], :current_confusion.shape[1]] = current_confusion
                total_confusion = new_total
                current_confusion = new_current
            total_confusion += current_confusion
        
        # 收集所有预测和真实标签
        all_preds.extend([p.flatten() for p in pre_label])
        all_labels.extend([l.flatten() for l in true_label])
        
        # 计算当前批次的评估指标
        eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        all_pixel_accuracies.append(eval_metrix['pixel_accuracy'])
        all_class_accuracies.append(eval_metrix['mean_class_accuracy'])

    # 使用总混淆矩阵计算最终的IoU值 - 修复的核心部分
    valid_samples = len(test_data) - error
    
    if total_confusion is not None:
        # 使用总混淆矩阵计算IoU
        final_iou = calc_semantic_segmentation_iou(total_confusion)
        final_miou = np.nanmean(final_iou)
        
        # 计算像素精度和类别精度
        final_pixel_accuracy = np.diag(total_confusion).sum() / total_confusion.sum()
        class_accuracy = np.diag(total_confusion) / (np.sum(total_confusion, axis=1) + 1e-10)
        final_class_accuracy = np.nanmean(class_accuracy)
        
        # 获取类别名称
        class_names = getattr(cfg, 'CLASS_NAMES', [f'类别 {i}' for i in range(num_class)])
        if len(class_names) > num_class:
            class_names = class_names[:num_class]
        elif len(class_names) < num_class:
            class_names.extend([f'类别 {i}' for i in range(len(class_names), num_class)])
        
        # 构建各类别IoU的输出字符串
        class_iou_str = "\n=== 各类别交并比（IoU） ===\n"
        for i in range(len(final_iou)):
            if i < len(class_names):
                class_iou_str += f"{class_names[i]}: {final_iou[i]:.5f}\n"
            else:
                class_iou_str += f"类别 {i}: {final_iou[i]:.5f}\n"
        
        epoch_str = ('test_acc :{:.5f} ,test_miou:{:.5f}, test_mpa:{:.5f}'.format(final_class_accuracy, final_miou, final_pixel_accuracy))
        
        # 新增：基于混淆矩阵计算各类别的精确率、召回率、准确率和F1
        tp = np.diag(total_confusion).astype(np.float64)
        row_sum = np.sum(total_confusion, axis=1).astype(np.float64)
        col_sum = np.sum(total_confusion, axis=0).astype(np.float64)
        fp = col_sum - tp
        fn = row_sum - tp
        tn = total_confusion.sum().astype(np.float64) - (tp + fp + fn)
        
        precision_per_class = np.zeros_like(tp, dtype=np.float64)
        recall_per_class = np.zeros_like(tp, dtype=np.float64)
        accuracy_per_class = np.zeros_like(tp, dtype=np.float64)
        f1_per_class = np.zeros_like(tp, dtype=np.float64)
        
        prec_den = tp + fp
        rec_den = tp + fn
        
        valid_prec = prec_den > 0
        valid_rec = rec_den > 0
        
        precision_per_class[valid_prec] = tp[valid_prec] / prec_den[valid_prec]
        recall_per_class[valid_rec] = tp[valid_rec] / rec_den[valid_rec]
        # per-class accuracy = (TP + TN) / total pixels
        total_pixels = total_confusion.sum().astype(np.float64)
        if total_pixels > 0:
            accuracy_per_class = (tp + tn) / total_pixels
        
        den = precision_per_class + recall_per_class
        valid_f1 = den > 0
        f1_per_class[valid_f1] = 2 * precision_per_class[valid_f1] * recall_per_class[valid_f1] / den[valid_f1]
        
        macro_precision = np.nanmean(precision_per_class)
        macro_recall = np.nanmean(recall_per_class)
        macro_accuracy = np.nanmean(accuracy_per_class)
        macro_f1 = np.nanmean(f1_per_class)
        
        # 微平均（micro）：基于总体汇总
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        micro_precision = tp_sum / (tp_sum + fp_sum + 1e-10)
        micro_recall = tp_sum / (tp_sum + fn_sum + 1e-10)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10)
        
        # 构建各类别指标输出字符串
        class_metrics_str = "\n=== 各类别指标（Precision/Recall/Accuracy/F1） ===\n"
        for i in range(len(tp)):
            name = class_names[i] if i < len(class_names) else f'类别 {i}'
            class_metrics_str += (f"{name}: 精确率={precision_per_class[i]:.5f}, "
                                  f"召回率={recall_per_class[i]:.5f}, "
                                  f"准确率={accuracy_per_class[i]:.5f}, "
                                  f"F1={f1_per_class[i]:.5f}\n")
        summary_metrics_str = ("\n=== 指标汇总 ===\n"
                               f"宏平均 Precision={macro_precision:.5f}, Recall={macro_recall:.5f}, "
                               f"Accuracy={macro_accuracy:.5f}, F1={macro_f1:.5f}\n"
                               f"微平均 Precision={micro_precision:.5f}, Recall={micro_recall:.5f}, "
                               f"F1={micro_f1:.5f}\n")
        
        # 打印测试结果、各类别IoU和新增指标
        print(epoch_str)
        print(class_iou_str)
        print(class_metrics_str)
        print(summary_metrics_str)
        
        # 保存指标到文件
        metrics_dir = './Results/metrics/'
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f'{model_name}_metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(epoch_str + "\n")
            f.write(class_iou_str + "\n")
            f.write(class_metrics_str + "\n")
            f.write(summary_metrics_str + "\n")
        print(f"详细指标已保存至: {metrics_path}")
        
        if final_miou > max(miou_list):
            miou_list.append(final_miou)
            print(epoch_str + '==========last')
        
        # 显示总混淆矩阵
        print("\n=== 混淆矩阵 ===")
        print(total_confusion)
        
        # 计算混淆矩阵的百分比 - 按行归一化
        row_sums = np.sum(total_confusion, axis=1, keepdims=True)
        # 处理除零错误
        row_sums = np.where(row_sums == 0, 1, row_sums)
        confusion_percent = total_confusion / row_sums * 100
        
        print("\n=== 混淆矩阵（按行归一化百分比）===")
        print(np.round(confusion_percent, 2))
        
        try:
            # 使用seaborn绘制混淆矩阵热力图
            plt.figure(figsize=(10, 8))
            # 设置中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            # 获取类别名称
            class_names = getattr(cfg, 'CLASS_NAMES', [f'类别 {i}' for i in range(num_class)])
            if len(class_names) != total_confusion.shape[0]:
                class_names = [f'类别 {i}' for i in range(total_confusion.shape[0])]
            sn.heatmap(total_confusion, annot=True, fmt="d", cmap="Blues",
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('语义分割混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            
            # 保存混淆矩阵图像
            save_dir = './Results/confusion_matrix/'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(save_path)
            print(f"混淆矩阵图像已保存至: {save_path}")
            
            # 显示混淆矩阵
            plt.show()
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")