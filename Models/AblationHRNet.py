import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.HRNet import HRNet


class MultiScaleSemanticBranch(nn.Module):
    """
    多尺度语义分支
    使用不同大小的卷积核来捕捉语义特征
    """

    def __init__(self, in_channels=64, out_channels=64):
        super(MultiScaleSemanticBranch, self).__init__()
        # 使用不同大小的卷积核来捕捉语义特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 注意力机制，增强语义特征的权重
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # HRNet输出是多尺度特征列表，我们取最高分辨率的特征
        if isinstance(features, list):
            x = features[0]
        else:
            x = features

        # 多尺度特征提取
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)

        # 特征融合
        x_fuse = torch.cat([x1, x3, x5], dim=1)
        x_fuse = self.fusion(x_fuse)

        # 注意力增强
        att = self.attention(x_fuse)
        x_att = x_fuse * att

        return x_att


class DilatedDetailBranch(nn.Module):
    """
    空洞细节分支
    使用空洞卷积和池化操作来捕捉细节特征
    """

    def __init__(self, in_channels=64, out_channels=64):
        super(DilatedDetailBranch, self).__init__()
        # 空洞卷积层，保持感受野的同时不降低分辨率
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 池化操作捕捉大尺度特征
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 注意力机制，增强细节特征的权重
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # HRNet输出是多尺度特征列表，我们取最高分辨率的特征
        if isinstance(features, list):
            x = features[0]
        else:
            x = features

        # 多尺度空洞卷积特征提取
        x1 = self.dilated_conv1(x)
        x2 = self.dilated_conv2(x)
        x3 = self.dilated_conv3(x)

        # 池化特征
        x_pool = self.pool(x)
        x_pool = self.pool_conv(x_pool)

        # 特征融合
        x_fuse = torch.cat([x1, x2, x3, x_pool], dim=1)
        x_fuse = self.fusion(x_fuse)

        # 注意力增强
        att = self.attention(x_fuse)
        x_att = x_fuse * att

        return x_att


def adaptive_fusion(semantic_feat, detail_feat):
    """
    自适应融合语义特征和细节特征
    根据不同区域的特征重要性动态调整权重
    """
    # 计算特征重要性
    semantic_importance = torch.sum(torch.abs(semantic_feat), dim=1, keepdim=True)
    detail_importance = torch.sum(torch.abs(detail_feat), dim=1, keepdim=True)

    # 归一化权重
    total_importance = semantic_importance + detail_importance + 1e-8  # 避免除零
    semantic_weight = semantic_importance / total_importance
    detail_weight = detail_importance / total_importance

    # 自适应加权融合
    fused_feat = semantic_feat * semantic_weight + detail_feat * detail_weight

    return fused_feat


class HRNetW48Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(HRNetW48Backbone, self).__init__()
        # 创建HRNet W48实例
        self.hrnet = HRNet(num_classes=3, pretrained=pretrained, width=48)
        # 保存原始的head，用于后续恢复
        self.original_head = self.hrnet.head
        
    def forward(self, x):
        # 直接访问HRNet内部方法，获取多尺度特征
        # stem
        x = self.hrnet.conv1(x)
        x = self.hrnet.bn1(x)
        x = self.hrnet.relu(x)
        x = self.hrnet.conv2(x)
        x = self.hrnet.bn2(x)
        x = self.hrnet.relu(x)

        # stage 1
        x = self.hrnet.layer1(x)
        x_list = []
        # 确保x_list的长度与transition1的数量匹配
        for i in range(len(self.hrnet.transition1)):
            if self.hrnet.transition1[i] is not None:
                x_list.append(self.hrnet.transition1[i](x))
            else:
                x_list.append(x)

        # stage 2
        x_list = self.hrnet.stage2(x_list)

        # stage 3
        # 正确处理transition2模块列表
        new_x_list = []
        for i in range(len(self.hrnet.transition2)):
            if i < len(x_list):
                if self.hrnet.transition2[i] is not None:
                    new_x_list.append(self.hrnet.transition2[i](x_list[i]))
                else:
                    new_x_list.append(x_list[i])
            else:
                # 处理新增的分支
                if self.hrnet.transition2[i] is not None:
                    # 使用最深层的特征来创建新分支
                    new_x_list.append(self.hrnet.transition2[i](x_list[-1]))
                else:
                    # 这种情况理论上不会发生
                    new_x_list.append(x_list[-1])
        x_list = new_x_list
        x_list = self.hrnet.stage3(x_list)

        # stage 4
        # 正确处理transition3模块列表
        new_x_list = []
        for i in range(len(self.hrnet.transition3)):
            if i < len(x_list):
                if self.hrnet.transition3[i] is not None:
                    new_x_list.append(self.hrnet.transition3[i](x_list[i]))
                else:
                    new_x_list.append(x_list[i])
            else:
                # 处理新增的分支
                if self.hrnet.transition3[i] is not None:
                    # 使用最深层的特征来创建新分支
                    new_x_list.append(self.hrnet.transition3[i](x_list[-1]))
                else:
                    # 这种情况理论上不会发生
                    new_x_list.append(x_list[-1])
        x_list = new_x_list
        x_list = self.hrnet.stage4(x_list)
        
        # 返回多尺度特征列表
        return x_list


class HRNetWithSemanticBranch(nn.Module):
    """
    原模型 + 模块一（多尺度语义分支）
    """

    def __init__(self, num_classes=3, pretrained=False):
        super(HRNetWithSemanticBranch, self).__init__()
        # 标准HRNet W48主干
        self.backbone = HRNetW48Backbone(pretrained=pretrained)

        # 只使用语义分支
        self.semantic_branch = MultiScaleSemanticBranch(in_channels=48, out_channels=64)  # 多尺度语义分支
        
        # 多尺度特征融合模块
        self.scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(48, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
        ])
        
        # 最终的分类层
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 获取输入图像尺寸，用于后续上采样
        h, w = x.size(2), x.size(3)

        # 通过主干网络提取多尺度特征
        multi_scale_features = self.backbone(x)
        
        # 提取最高分辨率特征用于语义分支
        high_res_feature = multi_scale_features[0]

        # 通过语义分支增强特征
        semantic_feat = self.semantic_branch(high_res_feature)
        
        # 分类预测
        output = self.classifier(semantic_feat)

        # 上采样到原始图像尺寸
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

        return output


class HRNetWithDetailBranch(nn.Module):
    """
    原模型 + 模块二（空洞细节分支）
    """

    def __init__(self, num_classes=3, pretrained=False):
        super(HRNetWithDetailBranch, self).__init__()
        # 标准HRNet W48主干
        self.backbone = HRNetW48Backbone(pretrained=pretrained)

        # 只使用细节分支
        self.detail_branch = DilatedDetailBranch(in_channels=48, out_channels=64)  # 空洞细节分支
        
        # 多尺度特征融合模块
        self.scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(48, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
        ])
        
        # 最终的分类层
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 获取输入图像尺寸，用于后续上采样
        h, w = x.size(2), x.size(3)

        # 通过主干网络提取多尺度特征
        multi_scale_features = self.backbone(x)
        
        # 提取最高分辨率特征用于细节分支
        high_res_feature = multi_scale_features[0]

        # 通过细节分支增强特征
        detail_feat = self.detail_branch(high_res_feature)
        
        # 分类预测
        output = self.classifier(detail_feat)

        # 上采样到原始图像尺寸
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

        return output


class HRNetWithSimpleFusion(nn.Module):
    """
    原模型 + 二者简单融合（相加）
    """

    def __init__(self, num_classes=3, pretrained=False):
        super(HRNetWithSimpleFusion, self).__init__()
        # 标准HRNet W48主干
        self.backbone = HRNetW48Backbone(pretrained=pretrained)

        # 语义和细节分支
        self.semantic_branch = MultiScaleSemanticBranch(in_channels=48, out_channels=64)  # 多尺度语义分支
        self.detail_branch = DilatedDetailBranch(in_channels=48, out_channels=64)  # 空洞细节分支
        
        # 简单融合后的处理
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 最终的分类层
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 获取输入图像尺寸，用于后续上采样
        h, w = x.size(2), x.size(3)

        # 通过主干网络提取多尺度特征
        multi_scale_features = self.backbone(x)
        
        # 提取最高分辨率特征用于语义-细节分支
        high_res_feature = multi_scale_features[0]

        # 通过语义-细节分支增强特征
        semantic_feat = self.semantic_branch(high_res_feature)
        detail_feat = self.detail_branch(high_res_feature)

        # 简单融合（相加）
        fused_feat = semantic_feat + detail_feat
        fused_feat = self.fusion_conv(fused_feat)
        
        # 分类预测
        output = self.classifier(fused_feat)

        # 上采样到原始图像尺寸
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

        return output


if __name__ == "__main__":
    # 测试模型
    print("测试 HRNetWithSemanticBranch...")
    model1 = HRNetWithSemanticBranch(num_classes=3)
    input_tensor = torch.randn(1, 3, 512, 512)
    output1 = model1(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output1.shape)
    
    print("\n测试 HRNetWithDetailBranch...")
    model2 = HRNetWithDetailBranch(num_classes=3)
    output2 = model2(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output2.shape)
    
    print("\n测试 HRNetWithSimpleFusion...")
    model3 = HRNetWithSimpleFusion(num_classes=3)
    output3 = model3(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output3.shape)