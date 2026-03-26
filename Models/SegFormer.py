import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# SegFormer不同版本的配置参数
def get_segformer_config(model_name):
    # 标准SegFormer模型配置 (B0-B5)
    configs = {
        'B0': {
            'embed_dims': [32, 64, 160, 256],
            'num_heads': [1, 2, 5, 8],
            'mlp_ratios': [4, 4, 4, 4],
            'depths': [2, 2, 2, 2],
            'sr_ratios': [8, 4, 2, 1],
            'drop_path_rate': 0.1
        },
        'B1': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'mlp_ratios': [4, 4, 4, 4],
            'depths': [2, 2, 2, 2],
            'sr_ratios': [8, 4, 2, 1],
            'drop_path_rate': 0.1
        },
        'B2': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'mlp_ratios': [4, 4, 4, 4],
            'depths': [3, 4, 6, 3],
            'sr_ratios': [8, 4, 2, 1],
            'drop_path_rate': 0.1
        },
        'B3': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'mlp_ratios': [4, 4, 4, 4],
            'depths': [3, 4, 18, 3],
            'sr_ratios': [8, 4, 2, 1],
            'drop_path_rate': 0.1
        },
        'B4': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'mlp_ratios': [4, 4, 4, 4],
            'depths': [3, 8, 27, 3],
            'sr_ratios': [8, 4, 2, 1],
            'drop_path_rate': 0.1
        },
        'B5': {
            'embed_dims': [64, 128, 320, 512],
            'num_heads': [1, 2, 5, 8],
            'mlp_ratios': [4, 4, 4, 4],
            'depths': [3, 6, 40, 3],
            'sr_ratios': [8, 4, 2, 1],
            'drop_path_rate': 0.1
        },
    }
    return configs.get(model_name, configs['B2'])  # 默认返回B2配置

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, 
                                             embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size//4, patch_size=3, stride=2, in_chans=embed_dims[0], 
                                             embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size//8, patch_size=3, stride=2, in_chans=embed_dims[1], 
                                             embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=embed_dims[2], 
                                             embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer) 
            for i in range(depths[0])])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer) 
            for i in range(depths[1])])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer) 
            for i in range(depths[2])])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer) 
            for i in range(depths[3])])

        # norm
        self.norm1 = norm_layer(embed_dims[0])
        self.norm2 = norm_layer(embed_dims[1])
        self.norm3 = norm_layer(embed_dims[2])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x):
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x)
        x = self.norm1(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs = [x]

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x)
        x = self.norm2(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x)
        x = self.norm3(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x)
        x = self.norm4(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.mlp(x)

class SegFormerHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, dropout_ratio=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        
        # 通道转换
        self.convs = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channel, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio)
        )
        
        # 输出层
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        # 调整各层特征的通道数
        outs = []
        for i in range(len(features)):
            x = self.convs[i](features[i])
            # 上采样到相同的空间尺寸（最小特征图的4倍）
            x = F.interpolate(x, size=(features[0].shape[2] * 2, features[0].shape[3] * 2), 
                              mode='bilinear', align_corners=False)
            outs.append(x)
        
        # 特征融合
        fusion = torch.cat(outs, dim=1)
        fusion = self.fusion_conv(fusion)
        
        # 预测
        output = self.cls_seg(fusion)
        return output

class SegFormer(nn.Module):
    def __init__(self, num_classes=12, pretrained=False, embed_dim=768, model_name='B2'):
        super().__init__()
        
        # 获取对应版本的配置参数
        config = get_segformer_config(model_name)
        
        # 构建MixVisionTransformer作为backbone
        self.backbone = MixVisionTransformer(
            img_size=352,  # 输入图像大小
            in_chans=3,
            embed_dims=config['embed_dims'],
            num_heads=config['num_heads'],
            mlp_ratios=config['mlp_ratios'],
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=config['drop_path_rate'],
            depths=config['depths'],
            sr_ratios=config['sr_ratios']
        )
        
        # 特征融合模块
        self.decode_head = SegFormerHead(
            in_channels=config['embed_dims'],
            channels=256,
            num_classes=num_classes,
            dropout_ratio=0.1
        )
        
        # 保存模型名称用于后续参考
        self.model_name = model_name

    def forward(self, x):
        # 获取不同尺度的特征
        features = self.backbone(x)
        # 特征融合和预测
        output = self.decode_head(features)
        # 上采样到原始图像大小
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

# 为每个版本创建单独的类，方便直接使用
class SegFormerB0(SegFormer):
    def __init__(self, num_classes=12, pretrained=False):
        super().__init__(num_classes=num_classes, pretrained=pretrained, model_name='B0')

class SegFormerB1(SegFormer):
    def __init__(self, num_classes=12, pretrained=False):
        super().__init__(num_classes=num_classes, pretrained=pretrained, model_name='B1')

class SegFormerB2(SegFormer):
    def __init__(self, num_classes=12, pretrained=False):
        super().__init__(num_classes=num_classes, pretrained=pretrained, model_name='B2')

class SegFormerB3(SegFormer):
    def __init__(self, num_classes=12, pretrained=False):
        super().__init__(num_classes=num_classes, pretrained=pretrained, model_name='B3')

class SegFormerB4(SegFormer):
    def __init__(self, num_classes=12, pretrained=False):
        super().__init__(num_classes=num_classes, pretrained=pretrained, model_name='B4')

class SegFormerB5(SegFormer):
    def __init__(self, num_classes=12, pretrained=False):
        super().__init__(num_classes=num_classes, pretrained=pretrained, model_name='B5')

if __name__ == "__main__":
    import torch as t
    print('-----'*5)
    rgb = t.randn(1, 3, 352, 480)
    
    # 测试不同版本的SegFormer模型
    for model_name in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']:
        print(f"Testing SegFormer{model_name}...")
        if model_name == 'B0':
            net = SegFormerB0(num_classes=12)
        elif model_name == 'B1':
            net = SegFormerB1(num_classes=12)
        elif model_name == 'B2':
            net = SegFormerB2(num_classes=12)
        elif model_name == 'B3':
            net = SegFormerB3(num_classes=12)
        elif model_name == 'B4':
            net = SegFormerB4(num_classes=12)
        else:
            net = SegFormerB5(num_classes=12)
        
        out = net(rgb)
        print(f"Output shape: {out.shape}")
        
        # 计算参数量
        param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Number of parameters: {param_count/1e6:.2f}M")
        print('-----'*5)