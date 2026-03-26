import torch.nn as nn
import torch as t
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset

from evalution_segmentaion import eval_semantic_segmentation
from Models import  UNet, Deeplab_v3plus, SegFormer, HRNet, SDHRNet, SDUNet, SDDeepLabv3plus, AblationHRNet, SegNeXt
import cfg
import argparse

# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 解析命令行参数
parser = argparse.ArgumentParser(description='Semantic Segmentation Training')
parser.add_argument('--model', type=str, default='UNet',  choices=[ 'SDUNet','SDDeepLabv3plus', 'UNet', 'Deeplab_v3plus', 'SegFormer', 'HRNet','SDHRNet', 'HRNetWithSemanticBranch', 'HRNetWithDetailBranch', 'HRNetWithSimpleFusion', 'SegNeXt'],
                    help='选择要使用的模型')
# 新增早停相关参数
parser.add_argument('--patience', type=int, default=30, help='早停耐心（连续未提升的epoch数）')
parser.add_argument('--monitor', type=str, default='miou', choices=['miou','loss','acc'], help='早停监控指标')
parser.add_argument('--min_delta', type=float, default=0.0, help='最小提升阈值')

args = parser.parse_args()

# 从配置中获取类别数量
num_class = cfg.DATASET[1]

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)

# 根据命令行参数选择模型

if args.model == 'UNet':
    model = UNet.UNet(3, num_class)
    model_name = 'UNet'
elif args.model == 'Deeplab_v3plus':
    model = Deeplab_v3plus.DeepLabv3_plus(nInputChannels=3, n_classes=num_class, os=16, _print=True)
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
elif args.model == 'SDHRNet':
    model = SDHRNet.SDHRNet(num_classes=num_class)
    model_name = 'SDHRNet'
elif args.model == 'SDUNet':
    model = SDUNet.SDUNet(in_channel=3, out_channel=num_class)
    model_name = 'SDUNet'
elif args.model == 'SDDeepLabv3plus':
        model = SDDeepLabv3plus.SDDeepLabv3plus(n_classes=num_class)
        model_name = 'SDDeepLabv3plus'
elif args.model == 'HRNetWithSemanticBranch':
    model = AblationHRNet.HRNetWithSemanticBranch(num_classes=num_class)
    model_name = 'HRNetWithSemanticBranch'
elif args.model == 'HRNetWithDetailBranch':
    model = AblationHRNet.HRNetWithDetailBranch(num_classes=num_class)
    model_name = 'HRNetWithDetailBranch'
elif args.model == 'HRNetWithSimpleFusion':
    model = AblationHRNet.HRNetWithSimpleFusion(num_classes=num_class)
    model_name = 'HRNetWithSimpleFusion'
elif args.model == 'SegNeXt':
    model = SegNeXt.SegNeXTB(num_classes=num_class)
    model_name = 'SegNeXt'
    # 初始化模型权重
    model.init_weights()

model = model.to(device)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
 
print(f'使用模型: {model_name}')


def train(model, model_name):
    # 早停状态
    best_val = None
    patience_counter = 0
    net = model.train()
    # 记录总的开始时间
    total_start_time = datetime.now()
    # 记录每个epoch的耗时
    epoch_durations = []
    
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        # 记录当前epoch的开始时间
        epoch_start_time = datetime.now()
        
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 25 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.8

        train_loss = 0
        train_acc = 0
        train_miou = 0
        # 记录当前epoch中每个batch的耗时
        batch_durations = []
        
        # 训练批次
        for i, sample in enumerate(train_data):
            # 记录batch开始时间
            batch_start_time = datetime.now()
            
            # 载入数据
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))
            # 训练
            out = net(img_data)
            
            # 检查输出是否包含nan值
            if t.isnan(out).any():
                print(f"Epoch {epoch+1}, Batch {i+1}: Model output contains NaN values!")
                # 打印输出的统计信息
                print(f"Output stats: mean={out.mean().item()}, std={out.std().item()}, min={out.min().item()}, max={out.max().item()}")
                # 保存当前批次的数据，以便调试
                t.save(img_data, f"debug_input_{epoch}_{i}.pt")
                t.save(img_label, f"debug_label_{epoch}_{i}.pt")
                exit()
            
            # 处理BanNet返回两个输出的情况
            if isinstance(out, tuple):
                main_out, aux_out = out
                main_out = F.log_softmax(main_out, dim=1)
                aux_out = F.log_softmax(aux_out, dim=1)
                loss = criterion(main_out, img_label) + 0.5 * criterion(aux_out, img_label)  # 主损失 + 辅助损失
                out = main_out
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, img_label)
            
            # 检查损失是否为nan
            if t.isnan(loss):
                print(f"Epoch {epoch+1}, Batch {i+1}: Loss is NaN!")
                # 打印模型输出的统计信息
                print(f"Output stats before softmax: mean={net(img_data).mean().item()}, std={net(img_data).std().item()}")
                print(f"Output stats after softmax: mean={out.mean().item()}, std={out.std().item()}")
                # 保存当前批次的数据和模型权重，以便调试
                t.save(img_data, f"debug_input_{epoch}_{i}.pt")
                t.save(img_label, f"debug_label_{epoch}_{i}.pt")
                t.save(model.state_dict(), f"debug_model_{epoch}_{i}.pt")
                exit()
            optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪，防止梯度爆炸
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            # 移除对class_accuracy数组的累加，只使用平均准确率
            
            
            # 计算当前batch耗时
            batch_duration = (datetime.now() - batch_start_time).seconds
            batch_durations.append(batch_duration)
            
            # 计算剩余batch数量和预计剩余时间
            remaining_batches = len(train_data) - (i + 1)
            remaining_epochs = cfg.EPOCH_NUMBER - (epoch + 1)
            
            # 计算平均batch耗时
            avg_batch_time = sum(batch_durations) / len(batch_durations)
            
            # 计算预计剩余时间
            remaining_time_in_epoch = remaining_batches * avg_batch_time
            
            # 如果已经完成过至少一个epoch，使用历史epoch耗时来估算
            if epoch > 0:
                avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
                remaining_time_total = remaining_time_in_epoch + (remaining_epochs * avg_epoch_time)
            else:
                # 否则使用当前epoch的平均batch时间来估算
                remaining_time_total = remaining_time_in_epoch + (remaining_epochs * (sum(batch_durations) + remaining_time_in_epoch))
            
            # 转换为小时、分钟、秒
            h_total, remainder_total = divmod(int(remaining_time_total), 3600)
            m_total, s_total = divmod(remainder_total, 60)
            
            print('|batch[{}/{}]|batch_loss {: .8f}| 剩余时间约: {:.0f}h {:.0f}m {:.0f}s'.format(
                i + 1, len(train_data), loss.item(), h_total, m_total, s_total))

        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
        )

        # 记录当前epoch的耗时
        epoch_duration = (datetime.now() - epoch_start_time).seconds
        epoch_durations.append(epoch_duration)
        
        # 计算总耗时和剩余epoch的预计时间
        total_elapsed_time = (datetime.now() - total_start_time).seconds
        remaining_epochs = cfg.EPOCH_NUMBER - (epoch + 1)
        
        if len(epoch_durations) > 0:
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
            remaining_time = remaining_epochs * avg_epoch_time
        else:
            remaining_time = 0
        
        # 转换为小时、分钟、秒
        h_elapsed, remainder_elapsed = divmod(total_elapsed_time, 3600)
        m_elapsed, s_elapsed = divmod(remainder_elapsed, 60)
        
        h_remaining, remainder_remaining = divmod(int(remaining_time), 3600)
        m_remaining, s_remaining = divmod(remainder_remaining, 60)
        
        print(metric_description)
        print(f'当前epoch耗时: {h_elapsed:.0f}h {m_elapsed:.0f}m {s_elapsed:.0f}s, 预计剩余时间: {h_remaining:.0f}h {m_remaining:.0f}m {s_remaining:.0f}s')
        
        # 验证评估
        val_loss, val_acc, val_miou = evaluate(model)
        # 选择监控指标
        if args.monitor == 'loss':
            current_val = val_loss
            is_improved = (best_val is None) or ((best_val - current_val) > args.min_delta)
        elif args.monitor == 'acc':
            current_val = val_acc
            is_improved = (best_val is None) or ((current_val - best_val) > args.min_delta)
        else:
            current_val = val_miou
            is_improved = (best_val is None) or ((current_val - best_val) > args.min_delta)
        
        # 保存最好模型并处理早停
        if is_improved:
            best_val = current_val
            patience_counter = 0
            import os
            save_dir = f'./Results/weights/{model_name}_weight/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            t.save(net.state_dict(), f'{save_dir}{epoch}.pth')
            print(f'验证集指标提升，已保存权重到: {save_dir}{epoch}.pth (monitor={args.monitor}, value={current_val:.5f})')
        else:
            patience_counter += 1
            print(f'验证集指标未提升 (monitor={args.monitor}, value={current_val:.5f})，耐心计数: {patience_counter}/{args.patience}')
            if patience_counter >= args.patience:
                print('触发早停，结束训练。')
                break


def evaluate(model):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0

    prec_time = datetime.now()
    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    avg_loss = eval_loss / len(val_data)
    avg_acc = eval_acc / len(val_data)
    avg_miou = eval_miou / len(val_data)
    
    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n'.format(
        avg_loss,
        avg_acc,
        avg_miou))
    print(val_str)
    print(time_str)
    
    # 返回指标用于早停判断
    return avg_loss, avg_acc, avg_miou


if __name__ == "__main__":
    train(model, model_name)
