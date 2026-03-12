import torch
import torch.nn as nn
from model.base_layer import Res_block


def _make_layer(block, input_channels, output_channels, num_blocks=2, downsample=False):
    layers = []
    if downsample:
        layers.append(block(input_channels, output_channels, stride=2))
    else:
        layers.append(block(input_channels, output_channels, stride=1))
    for _ in range(num_blocks - 1):
        layers.append(block(output_channels, output_channels))
    return nn.Sequential(*layers)

class ImageEncoder(nn.Module):
    def __init__(
        self,
        sam_encoder: nn.Module,
        _block: nn.Module = Res_block,
        backbone_channel_list: list[int] = [384, 192, 96],
        down_times: int = 3,
        stages: list[int] = [1, 2, 7],
    ):
        super().__init__()
        self.stages = stages

        self.channel_gen = []
        self.inc0 = nn.ModuleList()
        for i in range(down_times - 1, -1, -1):
            input_dim = 3 if i == down_times - 1 else backbone_channel_list[-1] // (2 ** (i + 1))
            out_dim = backbone_channel_list[-1] // (2**i)
            self.channel_gen.append((input_dim, out_dim))
            self.inc0.append(_make_layer(_block, input_dim, out_dim))

        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.stage_begin = [sum(stages[:i-1]) for i in range(1, len(stages) + 1)]

        self.incs = nn.ModuleList()
        embed_dim = backbone_channel_list[-1]
        for i in range(len(stages) - 1):
            dim_out = embed_dim * 2
            self.incs.append(_make_layer(_block, embed_dim, dim_out, num_blocks=1))
            embed_dim = dim_out

        for i in range(len(backbone_channel_list) - 1, 0, -1):
            self.channel_gen.append((backbone_channel_list[i], backbone_channel_list[i - 1]))
        self.channel_gen = self.channel_gen[1:]

        self.pool = nn.MaxPool2d(2, 2)

        self.trunk = sam_encoder

    def _process_initial_layers(self, x: torch.Tensor):
        """Process initial layers and store outputs."""
        out = x
        out_feats = []
        for blk in self.inc0:
            out = blk(out) if len(out_feats) == 0 else blk(self.pool(out))
            out_feats.append(out)
        return out_feats

    def _embed_and_position(self, x: torch.Tensor, ecd_embed: nn.Module):
        """Embed input and add positional encoding."""
        sam_out = ecd_embed(x)
        return sam_out + self.trunk._get_pos_embed(sam_out.shape[1:3])

    def forward(self, x: torch.tensor):
        pmt_blocks = self.trunk.promote_genertor.blocks
        ecd_embed = self.trunk.patch_embed
        ecd_blocks = self.trunk.blocks

        out_feats = self._process_initial_layers(x)
        out = out_feats[-1]

        sam_out = self._embed_and_position(x, ecd_embed)
        pmt_out = None

        inc_num = 0
        for i, sam_block in enumerate(ecd_blocks):
            if i in self.stage_ends[1:]:
                sam_out = sam_block(pmt_out + sam_out)
                out = self.incs[inc_num](self.pool(out)) + sam_out.permute(0, 3, 1, 2).contiguous()
                inc_num += 1
            else:
                sam_out = sam_block(sam_out)

            if i in self.stage_ends[:-1]:
                pmt_out = pmt_blocks[inc_num](sam_out + out.permute(0, 2, 3, 1).contiguous())

            if i in self.stage_ends[1:]:
                out_feats.append(out)

        return {
            "sam_backbone_embeds": out_feats[3: ],
            "dense_embeds": out_feats[: 3],
        }