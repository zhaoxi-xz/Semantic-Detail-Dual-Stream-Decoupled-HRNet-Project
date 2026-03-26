import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.SDHRNet import MultiScaleSemanticBranch, DilatedDetailBranch

# 初始化权重函数
def weights_init_normal(m):
    classname = m.__class__.__name__
    # 检查是否是Conv2d或ConvTranspose2d层
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # 确保卷积层有bias时也初始化
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    # 检查是否是BatchNorm2d层
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# 定义基本的卷积块，包含两层卷积、ReLU和BatchNorm
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.block(x)

# 定义上采样和跳跃连接的卷积块
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpConvBlock, self).__init__()
        # 上采样操作，将输入通道数减少到out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # ConvBlock的输入通道数应该是跳跃连接的通道数 + 上采样后的通道数
        self.block = ConvBlock(skip_channels + out_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1: 来自上一层的特征，x2: 跳跃连接的特征
        x1 = self.up(x1)
        
        # 计算需要填充的差值
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 连接跳跃连接的特征
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)

# 定义SDUNet模型
class SDUNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3):
        super(SDUNet, self).__init__()
        
        # 编码器部分 - 调整通道数和池化策略，避免特征图过小
        self.conv_encode1 = ConvBlock(in_channel, 64)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = ConvBlock(64, 128)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = ConvBlock(128, 256)
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = ConvBlock(256, 512)
        # 移除第4个池化，避免特征图太小
        
        # 瓶颈层 - 减小通道数，避免过拟合和内存问题
        self.bottleneck = ConvBlock(512, 512)
        
        # 解码器部分 - 使用新的UpConvBlock接口，指定跳跃连接通道数
        # in_channels, skip_channels, out_channels
        self.conv_decode4 = UpConvBlock(512, 512, 256)  # 512来自bottleneck, 512来自encode_block4, 输出256
        self.conv_decode3 = UpConvBlock(256, 256, 128)  # 256来自上一层, 256来自encode_block3, 输出128
        self.conv_decode2 = UpConvBlock(128, 128, 64)   # 128来自上一层, 128来自encode_block2, 输出64
        self.conv_decode1 = UpConvBlock(64, 64, 64)     # 64来自上一层, 64来自encode_block1_enhanced, 输出64
        
        # 语义-细节分支
        self.semantic_branch = MultiScaleSemanticBranch(64, 64)
        self.detail_branch = DilatedDetailBranch(64, 64)
        # 使用更完善的融合方式
        self.shape_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        
        # 最终输出层 - 移除ReLU，适合分割任务的输出
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
            # 移除ReLU，让softmax或损失函数处理非线性
        )
        
        # 应用权重初始化
        self.apply(weights_init_normal)
        
    def forward(self, x):
        # 记录输入尺寸，用于最终输出调整
        input_size = x.size()[2:]
        
        # 编码器路径
        encode_block1 = self.conv_encode1(x)
        
        # 语义-细节分支处理
        semantic_feat = self.semantic_branch(encode_block1)
        detail_feat = self.detail_branch(encode_block1)
        
        # 移除冗余计算，直接使用concat融合
        fused_shape = self.shape_fusion(torch.cat([semantic_feat, detail_feat], dim=1))
        
        # 将形状特征与原始特征相加（残差连接）
        encode_block1_enhanced = encode_block1 + fused_shape
        
        # 调整编码器路径，减少池化次数
        encode_pool1 = self.conv_pool1(encode_block1_enhanced)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        
        # 瓶颈层
        bottleneck = self.bottleneck(encode_block4)
        
        # 解码器路径 - 调整对应的跳跃连接
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1_enhanced)
        
        # 最终输出
        final = self.final_layer(decode_block1)
        
        # 动态调整输出尺寸，与输入保持一致
        final = F.interpolate(final, size=input_size, mode='bilinear', align_corners=True)
        
        return final


if __name__ == "__main__":
    import torch as t

    rgb = t.randn(1, 3, 572, 572)

    net = SDUNet(3, 12)

    out = net(rgb)

    print(out.shape)