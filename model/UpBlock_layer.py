import torch
import torch.nn as nn
import torch.nn.functional as F


class SE_Operate(nn.Module):
    def __init__(self, in_channels: int, out_channel: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channel, bias=True),
        )

    def forward(self, x: torch.tensor, att_x: torch.tensor):
        b, c, _, _ = x.size()
        y = F.avg_pool2d(att_x, (att_x.size(2), att_x.size(3)), stride=(att_x.size(2), att_x.size(3))).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + y.expand_as(x)


class CrossAtt(nn.Module):
    def __init__(self, in_channels: int, out_channel: int, MC=True):
        super().__init__()
        self.se_skip_att = SE_Operate(in_channels, out_channel)
        self.se_input_att = SE_Operate(in_channels, out_channel)
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.MC = MC
        # self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.tensor, skip_x: torch.tensor):
        x_up = self.up_sample(x)
        if self.MC:
            x_up_att = self.se_input_att(x_up, skip_x)
            x_skip_att = self.se_skip_att(skip_x, x)
        else:
            x_up_att = x_up
            x_skip_att = skip_x
        return torch.cat([x_up_att, x_skip_att], dim=1)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nb_Conv: int = 2, MC=True):
        super().__init__()
        self.cross_att = CrossAtt(in_channels=in_channels, out_channel=in_channels, MC=MC)

        # self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv)
        self.nConvs = _make_nConv(in_channels * 2, out_channels, nb_Conv)



    def forward(self, x: torch.tensor, skip_x: torch.tensor):
        out = self.cross_att(x, skip_x)
        return self.nConvs(out)


def _make_nConv(in_channels: int, out_channels: int, nb_Conv: int):
    layers = []
    layers.append(CBN(in_channels, out_channels))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.tensor):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


if __name__ == "__main__":
    x = torch.rand(3, 256, 64, 64)
    skip_x = torch.rand(3, 256, 128, 128)
    up_layer = UpBlock_attention(256 * 2, 128, nb_Conv=2)
    y = up_layer(x, skip_x)
    print(y.shape)
