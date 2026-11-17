import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class SpatialChannelAttention(nn.Module):

    def __init__(self, src_channels: int, tgt_channels: int,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

   
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(src_channels, tgt_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(tgt_channels // 4, tgt_channels),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(src_channels, src_channels,
                      kernel_size=3, padding=1,
                      groups=src_channels, bias=False),  
            nn.Conv2d(src_channels, 1,
                      kernel_size=1, bias=False),         
            nn.Sigmoid()
        )

        self.refine = ConvModule(
            tgt_channels, tgt_channels,
            kernel_size=3, padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        src: [B, C_src, H_src, W_src]  深层特征，提供语义先验
        tgt: [B, C_tgt, H_tgt, W_tgt]  浅层特征，被增强
        """
        ch_attn = self.channel_attn(src)                    
        ch_attn = ch_attn.view(-1, tgt.shape[1], 1, 1)     

        src_up = F.interpolate(src, size=tgt.shape[2:],
                               mode='bilinear', align_corners=False)
        sp_attn = self.spatial_attn(src_up)                

        gate = ch_attn * sp_attn                          
        enhanced = self.refine(tgt * gate) + tgt

        return enhanced


class FAM(nn.Module):
    """Flood-Aware Multi-scale Aggregation Neck（修复版）.

    Args:
        in_channels (list[int]): encoder 各阶段输出通道
            MiT-B2: [64, 128, 320, 512]
        norm_cfg (dict): BN 配置
    """

    def __init__(self, in_channels: list,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.in_channels = in_channels
        n = len(in_channels)  

        self.top_down = nn.ModuleList([
            SpatialChannelAttention(
                src_channels=in_channels[i + 1],
                tgt_channels=in_channels[i],
                norm_cfg=norm_cfg
            )
            for i in range(n - 1)   
        ])

        self.bottom_up = nn.ModuleList([
            ConvModule(
                in_channels[i], in_channels[i + 1],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')
            )
            for i in range(n - 1)   
        ])

    def forward(self, inputs: list) -> list:
        """
        inputs: [F1(1/4), F2(1/8), F3(1/16), F4(1/32)]
        returns: 同尺寸同通道，特征被增强后的四个尺度
        """
        f1, f2, f3, f4 = inputs

        f3 = self.top_down[2](src=f4, tgt=f3)
        f2 = self.top_down[1](src=f3, tgt=f2)
        f1 = self.top_down[0](src=f2, tgt=f1)

        f2 = f2 + self.bottom_up[0](F.avg_pool2d(f1, kernel_size=2, stride=2))
        f3 = f3 + self.bottom_up[1](F.avg_pool2d(f2, kernel_size=2, stride=2))
        f4 = f4 + self.bottom_up[2](F.avg_pool2d(f3, kernel_size=2, stride=2))

        return [f1, f2, f3, f4]

