import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method

        self._check_branches(blocks, num_blocks, num_inchannels, num_channels)

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(self, blocks, num_blocks, num_inchannels, num_channels):
        if self.num_branches != len(num_blocks):
            error_msg = 'num_branches({}) != len(num_blocks)({})'
            raise ValueError(error_msg.format(self.num_branches, len(num_blocks)))

        if self.num_branches != len(num_channels):
            error_msg = 'num_branches({}) != len(num_channels)({})'
            raise ValueError(error_msg.format(self.num_branches, len(num_channels)))

        if self.num_branches != len(num_inchannels):
            error_msg = 'num_branches({}) != len(num_inchannels)({})'
            raise ValueError(error_msg.format(self.num_branches, len(num_inchannels)))

    def _make_one_branch(self, block, num_blocks, num_channels, in_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != num_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, num_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, num_channels, stride, downsample))
        current_inchannels = num_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(current_inchannels, num_channels))

        return nn.Sequential(*layers), current_inchannels

    def _make_branches(self, blocks, num_blocks, num_channels):
        branches = []
        # 创建一个副本以避免修改原始数组
        new_num_inchannels = self.num_inchannels.copy()

        for i in range(self.num_branches):
            branch, new_inchannels = self._make_one_branch(blocks[i], num_blocks[i], num_channels[i], new_num_inchannels[i])
            branches.append(branch)
            new_num_inchannels[i] = new_inchannels
            
        # 更新num_inchannels
        self.num_inchannels = new_num_inchannels
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                  kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='bilinear', align_corners=True)
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(y)

        return x_fuse

class HRNet(nn.Module):
    def __init__(self, num_classes=12, pretrained=False, width=32):
        super(HRNet, self).__init__()
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 根据width参数设置通道数
        if width == 48:
            # HRNet W48配置
            stage1_channels = 96
            stage2_channels = [48, 96]
            stage3_channels = [48, 96, 192]
            stage4_channels = [48, 96, 192, 384]
        else:
            # 默认HRNet W32配置
            stage1_channels = 64
            stage2_channels = [32, 64]
            stage3_channels = [32, 64, 128]
            stage4_channels = [32, 64, 128, 256]

        # stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, stage1_channels, 4)
        self.stage1_cfg = {
            'num_branches': 1,
            'num_blocks': [4],
            'num_channels': [stage1_channels],
            'block': Bottleneck,
            'fuse_method': 'SUM'
        }
        self.transition1 = self._make_transition_layer([stage1_channels * 4], stage2_channels)
        self.stage2_cfg = {
            'num_branches': 2,
            'num_blocks': [4, 4],
            'num_channels': stage2_channels,
            'block': BasicBlock,
            'fuse_method': 'SUM'
        }
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, stage2_channels)

        # stage 3
        self.transition2 = self._make_transition_layer(pre_stage_channels, stage3_channels)
        self.stage3_cfg = {
            'num_branches': 3,
            'num_blocks': [4, 4, 4],
            'num_channels': stage3_channels,
            'block': BasicBlock,
            'fuse_method': 'SUM'
        }
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, stage3_channels)

        # stage 4
        self.transition3 = self._make_transition_layer(pre_stage_channels, stage4_channels)
        self.stage4_cfg = {
            'num_branches': 4,
            'num_blocks': [4, 4, 4, 4],
            'num_channels': stage4_channels,
            'block': BasicBlock,
            'fuse_method': 'SUM'
        }
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, stage4_channels, multi_scale_output=False)

        # segmentation head
        self.head = HRNetHead(in_channels=pre_stage_channels[0], out_channels=num_classes)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                  kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels,
                                  kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['num_modules'] if 'num_modules' in layer_config else 1
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = layer_config['block']
        fuse_method = layer_config['fuse_method']

        modules = []
        for i in range(num_modules):
            if i == 0 and not multi_scale_output:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(
                num_branches, 
                [block] * num_branches, 
                num_blocks, 
                num_inchannels, 
                num_channels, 
                fuse_method, 
                reset_multi_scale_output
            ))
            num_inchannels = modules[-1].num_inchannels

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # stage 1
        x = self.layer1(x)
        x_list = []
        # 确保x_list的长度与transition1的数量匹配
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # stage 2
        x_list = self.stage2(x_list)

        # stage 3
        # 正确处理transition2模块列表
        new_x_list = []
        for i in range(len(self.transition2)):
            if i < len(x_list):
                if self.transition2[i] is not None:
                    new_x_list.append(self.transition2[i](x_list[i]))
                else:
                    new_x_list.append(x_list[i])
            else:
                # 处理新增的分支
                if self.transition2[i] is not None:
                    # 使用最深层的特征来创建新分支
                    new_x_list.append(self.transition2[i](x_list[-1]))
                else:
                    # 这种情况理论上不会发生
                    new_x_list.append(x_list[-1])
        x_list = new_x_list
        x_list = self.stage3(x_list)

        # stage 4
        # 正确处理transition3模块列表
        new_x_list = []
        for i in range(len(self.transition3)):
            if i < len(x_list):
                if self.transition3[i] is not None:
                    new_x_list.append(self.transition3[i](x_list[i]))
                else:
                    new_x_list.append(x_list[i])
            else:
                # 处理新增的分支
                if self.transition3[i] is not None:
                    # 使用最深层的特征来创建新分支
                    new_x_list.append(self.transition3[i](x_list[-1]))
                else:
                    # 这种情况理论上不会发生
                    new_x_list.append(x_list[-1])
        x_list = new_x_list
        x_list = self.stage4(x_list)

        # segmentation head
        x = self.head(x_list[0])
        # 添加上采样操作，将输出特征图恢复到与输入图像相同的分辨率
        # 假设输入图像尺寸为512x512，当前输出为128x128，需要上采样4倍
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x

class HRNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HRNetHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    import torch as t
    print('-----'*5)
    rgb = t.randn(1, 3, 352, 480)
    
    net = HRNet(num_classes=12)
    out = net(rgb)
    
    print(out.shape)