import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    def __init__(self, stages: list[int], embed_dim: int = 96):
        super().__init__()
        self.stages = stages
        self.embed_dim = embed_dim
        self.embed_layer_num = [sum(stages[:i]) for i in range(1, len(stages) + 1)]
        self.patchEmded = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)),
        )
        self.blocks = nn.ModuleList()
        for _ in range(len(stages) - 1):
            dim_out = embed_dim * 2
            block = PmtConvBlock(embed_dim, dim_out, is_stride=True)
            embed_dim = dim_out

            self.blocks.append(block)

    def forward(self, x: torch.Tensor):  # B, C, H, W
        out_feat = []
        x = self.patchEmded(x).permute(0, 2, 3, 1).contiguous()
        for blk in self.blocks:
            x = blk(x)
            out_feat.append(x)
        return out_feat


class PmtConvBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int, is_stride: bool = False):
        super().__init__()
        if is_stride:
            self.blk = nn.Sequential(
                nn.Conv2d(input_ch, input_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(input_ch),
                nn.GELU(),
                nn.Conv2d(input_ch, output_ch, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(output_ch),
                nn.GELU(),
            )
        else:
            self.blk = nn.Sequential(
                nn.Conv2d(input_ch, input_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(input_ch),
                nn.GELU(),
                nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_ch),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor):  # B, H, W, C
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        x = self.blk(x)
        return x.permute(0, 2, 3, 1).contiguous()
        # return x

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiScalePositionalEncoder(nn.Module):
    def __init__(self, in_chans=[24, 48, 96], down_times=[5, 4, 3], embed_dim=256):
        super().__init__()

        if len(in_chans) != len(down_times):
            raise ValueError(f"MultiScalePositionalEncoder: in_chans {in_chans} must have the same length as down_times {down_times}")

        self.proj_blocks = nn.ModuleList()
        for in_ch, down_t in zip(in_chans, down_times):
            blk = nn.Sequential(
                *[ConvolutionalBlock(in_ch, in_ch, kernel_size=3, stride=2, padding=1) for _ in range(down_t)],
                nn.Conv2d(in_ch, embed_dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            self.proj_blocks.append(blk)

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_chans), embed_dim, 1, 1, bias=True, groups=embed_dim),
        )

    def forward(self, scale_feat: list[torch.tensor]):
        feat_tokens = []
        for feat, proj_block in zip(scale_feat, self.proj_blocks):
            feat_tokens.append(proj_block(feat))

        cnn_feat = torch.stack(feat_tokens, dim=1).flatten(start_dim=1, end_dim=2)
        x = self.proj(cnn_feat)
        return x
