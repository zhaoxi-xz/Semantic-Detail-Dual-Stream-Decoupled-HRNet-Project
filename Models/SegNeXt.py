import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. 核心算子：MSCA (Multi-Scale Convolutional Attention)
# --------------------------------------------------------------------------
class MSCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 局部卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 多尺度条带卷积分支
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # 通道混合
        self.conv_step2 = nn.Conv2d(dim, dim, 1)

        # 添加激活函数，增强数值稳定性
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        # 多尺度注意力计算，添加激活函数和clamp
        attn_1 = self.act(self.conv1_2(self.act(self.conv1_1(attn))))
        attn_2 = self.act(self.conv2_2(self.act(self.conv2_1(attn))))
        attn_3 = self.act(self.conv3_2(self.act(self.conv3_1(attn))))

        # 聚合注意力，添加clamp防止数值溢出
        attn = attn + attn_1 + attn_2 + attn_3
        attn = torch.clamp(attn, min=-10.0, max=10.0)  # 限制数值范围

        attn = self.conv_step2(attn)
        attn = torch.sigmoid(attn)  # 使用sigmoid确保注意力权重在0-1之间

        # 注意力加权
        return u * attn


# --------------------------------------------------------------------------
# 2. 基础组件：FFN 与 Block
# --------------------------------------------------------------------------
class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, stride=1, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SegNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MSCA(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


# --------------------------------------------------------------------------
# 3. 解码头
# --------------------------------------------------------------------------
class SegNeXtHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for in_ch in in_channels
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        outs = []
        for i, x in enumerate(inputs):
            x = self.convs[i](x)
            if i > 0:
                x = F.interpolate(x, size=inputs[0].shape[2:], mode='bilinear', align_corners=False)
            outs.append(x)
        out = self.fusion_conv(torch.cat(outs, dim=1))
        return self.cls_seg(out)


# --------------------------------------------------------------------------
# 4. 主类定义
# --------------------------------------------------------------------------
class SegNeXt(nn.Module):
    def __init__(self, num_classes=12, embed_dims=[64, 128, 320, 512], depths=[3, 3, 27, 3]):
        super().__init__()
        # Patch Embeddings
        self.patch_embed1 = OverlapPatchEmbed(7, 4, 3, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        # Stages
        self.stage1 = nn.Sequential(*[SegNeXtBlock(embed_dims[0]) for _ in range(depths[0])])
        self.stage2 = nn.Sequential(*[SegNeXtBlock(embed_dims[1]) for _ in range(depths[1])])
        self.stage3 = nn.Sequential(*[SegNeXtBlock(embed_dims[2]) for _ in range(depths[2])])
        self.stage4 = nn.Sequential(*[SegNeXtBlock(embed_dims[3]) for _ in range(depths[3])])

        self.decode_head = SegNeXtHead(embed_dims, 256, num_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        x1 = self.stage1(self.patch_embed1(x))
        x2 = self.stage2(self.patch_embed2(x1))
        x3 = self.stage3(self.patch_embed3(x2))
        x4 = self.stage4(self.patch_embed4(x3))

        out = self.decode_head([x1, x2, x3, x4])
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

    def init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------------
# 5. 专门用于实例化的包装类 (解决你的 AttributeError)
# --------------------------------------------------------------------------
class SegNeXTB(SegNeXt):
    def __init__(self, num_classes=12):
        # 对齐 SegNeXt-B 标准配置
        super().__init__(num_classes=num_classes, embed_dims=[64, 128, 320, 512], depths=[3, 3, 27, 3])


class SegNeXTT(SegNeXt):
    def __init__(self, num_classes=12):
        super().__init__(num_classes=num_classes, embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2])