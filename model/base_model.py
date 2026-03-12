"""Base model utilities for SAM-SPL.

This module provides adapter components that connect a SAM-style image encoder
with the project's custom decoder and mask heads. The key elements are:
- `SamAdaptor`: adapts SAM encoder outputs into multi-scale mask outputs.
- `DynamicConvBlock` and `build_dynamic_conv`: helpers to build dynamic
    convolutional blocks that optionally downsample based on stage count.

Only documentation strings have been added/updated; no computational logic
is changed by these edits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_layer import DenseBlock, SELayer, VGGBlock, Res_CBAM_block
from model.UpBlock_layer import UpBlock_attention
from model.hieradet import Hiera
from model.pmt_generator import MultiScaleBlock, MultiScalePositionalEncoder

from model.transformer import TwoWayTransformer
from model.utils import LayerNorm2d, MLP
from model.image_encoder import ImageEncoder
import logging
import math

class SinePositionalEncoding(nn.Module):
    """
    Helper: Converts (x, y) coordinates to Sine/Cosine embeddings.
    Standard for Transformer position encoding (DETR/SAM style).
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        # x: [B, N_queries, 2] -> containing (x, y) coordinates in [0, 1] range
        y_embed = x[:, :, 1]
        x_embed = x[:, :, 0]
        
        if self.normalize:
            # Assuming coords are already [0, 1], we scale them to 2*pi
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        # Interleave sine/cosine
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # Concatenate to get [B, N, num_pos_feats*2]
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos

class ContrastQueryGenerator(nn.Module):
    """
    Innovation Direction 3 (Enhanced): Energy-Weighted Global Saliency Aggregation (EWGSA)
    
    Instead of returning Top-K distinct queries, this module:
    1. Identifies Top-K high-frequency regions (internal candidates).
    2. Generates position embeddings for all candidates.
    3. Aggregates them into a SINGLE token using energy-based weighted summation.
    
    This forces the single token to represent the "global saliency center" of all potential targets.
    """
    def __init__(self, rhpm_energy, embed_dim, internal_k=20):
        super().__init__()
        self.internal_k = internal_k # 内部采样的点数，建议设置较大 (e.g., 20) 以覆盖所有潜在目标
        self.embed_dim = embed_dim
        
        # 1. RHPM module
        self.rhpm = AdaptiveRHPM(energy_threshold=rhpm_energy)
        
        # 2. Positional Encoding Helper
        self.pos_encoder = SinePositionalEncoding(num_pos_feats=embed_dim // 2)
        
        # 3. Fusion Layer: 
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 4. Content Embedding (Global Template)
        self.query_content_embed = nn.Embedding(1, embed_dim)

    def forward(self, feature_map):
        """
        Args:
            feature_map: [B, C, H, W]
        Returns:
            fused_token: [B, 1, C] - The single aggregated query token
        """
        B, C, H, W = feature_map.shape
        
        # --- Step 1: Extract High-Frequency Energy ---
        high_freq_map = self.rhpm(feature_map)
        energy_map = high_freq_map.mean(dim=1) # [B, H, W]
        flat_energy = energy_map.view(B, -1) # [B, H*W]
        
        # --- Step 2: Select Top-K Candidates (Internal) ---
        # values: 能量值 (用于加权), indices: 位置索引
        # 我们选取前 internal_k 个点，比如前 20 个最亮的点
        values, indices = torch.topk(flat_energy, k=self.internal_k, dim=1)
        
        # --- Step 3: Generate Coordinate Embeddings ---
        y_coords = (indices // W).float() / H
        x_coords = (indices % W).float() / W
        coords = torch.stack([x_coords, y_coords], dim=-1) # [B, internal_k, 2]
        
        # 生成 20 个位置的 Embedding
        # shape: [B, internal_k, embed_dim]
        raw_embeddings = self.pos_encoder(coords)
        
        # --- Step 4: Energy-Weighted Aggregation (The Innovation) ---
        
        # 计算权重：能量越大的点，权重越大
        # Softmax 确保权重之和为 1，实现了几何上的“重心”聚合
        weights = F.softmax(values, dim=1).unsqueeze(-1) # [B, internal_k, 1]
        
        # 加权求和：将 20 个向量融合成 1 个向量
        # shape: [B, 1, embed_dim]
        aggregated_pos = (raw_embeddings * weights).sum(dim=1, keepdim=True)
        
        # --- Step 5: Feature Refinement ---
        # 融合位置信息与可学习的内容信息
        # content_embed: [1, embed_dim] -> [B, 1, embed_dim]
        content = self.query_content_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # 将聚合的位置信息注入到内容 Token 中
        # 通过 Linear 层让特征更平滑
        fused_token = content + self.fusion_layer(aggregated_pos)
        
        return fused_token
        
class AdaptiveRHPM(nn.Module):
    """
    改进版动态高通滤波器：
    1. 采用矢量化搜索 (searchsorted) 替代 Python 循环。
    2. 采用圆形掩码 (Radial Mask) 替代 矩形掩码，符合频域物理特性。
    3. 支持多通道/批次并行计算。
    """
    def __init__(self, energy_threshold=0.8):
        super(AdaptiveRHPM, self).__init__()
        self.threshold = energy_threshold

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # --- 1. 频域转换 (使用正交归一化保持能量) ---
        f = torch.fft.fft2(x, norm='ortho')
        fshift = torch.fft.fftshift(f)
        
        # --- 2. 构造径向距离矩阵 (Normalized Distance Grid) ---
        # 预先生成坐标系，减少重复计算
        y = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x_coords, indexing='ij')
        dist = torch.sqrt(grid_y**2 + grid_x**2) # 中心距离矩阵 [H, W]
        dist_flat = dist.view(-1)
        sorted_dist, sort_indices = torch.sort(dist_flat)

        # --- 3. 动态搜索截止半径 ---
        # 计算功率谱密度 (PSD)
        magnitude = torch.abs(fshift)
        energy_map = magnitude ** 2
        # 按样本处理能量 (B, C, H, W) -> (B, H*W) 
        # 这里取所有通道的平均能量分布来确定截止半径
        energy_per_sample = energy_map.mean(dim=1).view(B, -1) 
        
        # 按照距离从小到大重排能量分布
        sorted_energy = energy_per_sample[:, sort_indices]
        # 计算累积能量百分比
        cum_energy = torch.cumsum(sorted_energy, dim=-1)
        total_energy = cum_energy[:, -1:]
        
        # 找到达到阈值的能量索引
        # 使用 searchsorted 快速定位每一个样本的截止位置
        target_energy = total_energy * self.threshold
        # 为每个 Batch 找到对应的截止半径索引
        cutoff_indices = torch.searchsorted(cum_energy, target_energy).clamp(0, H*W-1)
        # 获取具体的截止半径值
        cutoff_radii = sorted_dist[cutoff_indices] # [B, 1]

        # --- 4. 应用圆形掩码 ---
        # 构造掩码：距离大于半径的保留（高频），距离小于半径的置0（低频）
        # dist: [H, W], cutoff_radii: [B, 1, 1]
        mask = (dist.unsqueeze(0) > cutoff_radii.unsqueeze(-1)).float()
        
        # 应用掩码并还原
        fshift_filtered = fshift * mask.unsqueeze(1) # 广播到所有通道
        ishift = torch.fft.ifftshift(fshift_filtered)
        # 反变换回时域
        return torch.abs(torch.fft.ifft2(ishift))


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    try:
        if classname.find("Conv") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif classname.find("Linear") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    except AttributeError:
        pass

class DynamicConvBlock(nn.Module):
    """Dynamic convolutional block with optional downsampling stages.

    The block always starts with a 1x1 convolution + BatchNorm + GELU for
    channel fusion. Depending on the provided stage index ``n`` (must be
    ``<= 4``) the block will append up to ``4 - n`` 2x2 stride-2 convolutions
    each followed by GELU to perform additional downsampling.

    Parameters
    ----------
    skip_channel : int
        Number of input/output channels.
    n : int
        Stage index used to determine how many downsampling convolutions to
        append. Must satisfy ``n <= 4``.
    """
    def __init__(self, skip_channel, n):
        super().__init__()
        self.skip_channel = skip_channel
        if n > 4:
            raise ValueError(f"n must be <= 4, but got {n}")
        self.n = n
        self.conv_blocks = self._build_blocks()
    
    def _build_blocks(self):
        """Build and return the sequential convolutional layers.

        The returned module always begins with a 1x1 conv + BN + GELU and then
        contains (4 - n) blocks of 2x2 stride-2 conv + GELU.
        """
        layers = [
            nn.Conv2d(self.skip_channel, self.skip_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.skip_channel),
            nn.GELU()
        ]

        num_extra_blocks = 4 - self.n
        
        for _ in range(num_extra_blocks):
            layers.extend([
                nn.Conv2d(self.skip_channel, self.skip_channel, kernel_size=2, stride=2),
                nn.GELU()
            ])

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the constructed convolutional sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, H, W), where C == ``skip_channel``.

        Returns
        -------
        torch.Tensor
            Output tensor after the conv sequence.
        """
        return self.conv_blocks(x)
    
def build_dynamic_conv(skip_channel: int, n: int) -> nn.Sequential:
    """Functional helper that returns an ``nn.Sequential`` with the same
    rules as :class:`DynamicConvBlock`.

    Parameters
    ----------
    skip_channel : int
        Number of channels.
    n : int
        Stage index; controls how many downsampling convs are added.

    Returns
    -------
    nn.Sequential
        Sequential module implementing the dynamic conv pattern.
    """
    if n > 4:
        raise ValueError(f"n must be <= 4, but got {n}")
    layers = [
        nn.Conv2d(skip_channel, skip_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(skip_channel),
        nn.GELU()
    ]

    for _ in range(4 - n):
        layers.extend([
            nn.Conv2d(skip_channel, skip_channel, kernel_size=2, stride=2),
            nn.GELU()
        ])
    
    return nn.Sequential(*layers)

class SamAdaptor(nn.Module):
    """Adapter that converts SAM encoder outputs into multi-scale masks.

    The adapter fuses SAM multi-scale features with an auxiliary dense branch
    and optionally uses a transformer-based decoder (``decoder_transformer``)
    plus a hypernetwork to produce deep mask features. These features are
    then passed through upsampling blocks and skip connections to produce
    multi-scale mask tensors.

    See the constructor for parameter descriptions.
    """
    def __init__(
        self,
        sam_encoder: nn.Module,
        decoder_transformer: nn.Module,
        backbone_channel_list: list[int] = [384, 192, 96],
        stages=[1, 2, 7],
        block="res",
        dense_low_channels: list[int] = [96, 48, 24],
        num_mask_tokens=1,
        use_sam_decoder=True,
        pe_inch=[24, 48, 96],
        energy = [0.1, 0.2, 0.4, 0.8],
    ):
        super().__init__()
        self.use_sam_decoder = use_sam_decoder
        self.num_mask_tokens = num_mask_tokens
        self.dense_low_channels = backbone_channel_list + dense_low_channels[1:]
        self.pe_inch = pe_inch
        if block == "res":
            _block = Res_CBAM_block
        elif block == "vgg":
            _block = VGGBlock
        elif block == "dense":
            _block = DenseBlock

        _block = self._select_block(block)
        self.image_encoder = ImageEncoder(
            sam_encoder,
            _block=_block,
            backbone_channel_list=backbone_channel_list,
            stages=stages,
        )
        # HDNet-style initial feature map (x_py_init)
        self.hdnet_init_conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(3, 1, 3, 1, 1)
        
        self.py2 = AdaptiveRHPM(energy[2])
        self.py1 = AdaptiveRHPM(energy[1])
        self.py0 = AdaptiveRHPM(energy[0])
        self.sigmoid = nn.Sigmoid()
        self.skip_channel_gen = dense_low_channels
        self.mask_channel_gen = [ch // 2 for ch in dense_low_channels]
        if self.use_sam_decoder:
            self.up_decoders, self.skip_convs = self._initialize_up_decoders_and_skip_convs()
        else:
            self.up_decoders, self.skip_convs = self._initialize_up_decoders_and_skip_convs2()

        self.reduction_convs = self._initialize_reduction_convs()

        # Projection block to match the dimensions of the dense low channels
        if self.use_sam_decoder:
            self.decoder_dim = decoder_transformer.embedding_dim
            self.decoder_transformer = decoder_transformer
      
            self.query_generator = ContrastQueryGenerator(
                rhpm_energy=0.2, 
                embed_dim=self.decoder_dim,
                internal_k=10 
            )
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(self.decoder_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(self.decoder_dim // 4, dense_low_channels[0], kernel_size=2, stride=2),
                nn.GELU(),
            )
            self.output_hypernetworks_mlp = MLP(self.decoder_dim, self.decoder_dim, dense_low_channels[0], 3)
            self.image_pe_encoder = MultiScalePositionalEncoder(
                in_chans=pe_inch,
                down_times=[len(self.dense_low_channels) - i - 1 for i in range(len(pe_inch))],
            )
            self.proj_block = build_dynamic_conv(self.skip_channel_gen[0], len(stages))

            self.deep_conv_block = nn.Sequential(
                nn.Conv2d(backbone_channel_list[0], self.decoder_dim, kernel_size=1, stride=1),
                LayerNorm2d(self.decoder_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=1, stride=1),
                nn.GELU(),
            )
        else:
            self.proj_block = nn.Sequential(
                nn.Conv2d(self.dense_low_channels[0], self.dense_low_channels[1], kernel_size=1, stride=1),
                nn.BatchNorm2d(self.dense_low_channels[1]),
                nn.GELU(),
                nn.Conv2d(self.dense_low_channels[1], self.dense_low_channels[1], kernel_size=1, stride=1),
                nn.GELU(),
            )

        
        self.apply(weights_init_kaiming)

    def _select_block(self, block: str) -> nn.Module:
        """Select an encoder block class by name.

        Parameters
        ----------
        block : str
            One of 'res', 'vgg', or 'dense'.

        Returns
        -------
        nn.Module
            The block class corresponding to the provided name. Defaults to
            :class:`Res_CBAM_block` if an unknown name is given.
        """
        blocks = {"res": Res_CBAM_block, "vgg": VGGBlock, "dense": DenseBlock}
        return blocks.get(block, Res_CBAM_block)  # Default to Res_CBAM_block if not found

    def _initialize_up_decoders_and_skip_convs(self) -> tuple:
        """Initialize up-sampling decoder blocks and skip convolution modules.

        Returns
        -------
        tuple
            A tuple ``(up_decoders, skip_convs)`` where each element is an
            ``nn.ModuleList`` containing the corresponding modules.
        """

        up_decoders = nn.ModuleList()
        skip_convs = nn.ModuleList()
        for in_ch in self.skip_channel_gen:
            up_decoders.append(UpBlock_attention(in_ch, in_ch // 2))
            skip_convs.append(
                nn.Sequential(
                    SELayer(in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.GELU(),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            )

        return up_decoders, skip_convs
        
   
    def _initialize_up_decoders_and_skip_convs2(self) -> tuple:
        """Variant initialization that uses ``self.dense_low_channels[1:]``
        as the input channel list for skip modules.

        Returns
        -------
        tuple
            ``(up_decoders, skip_convs)`` as ``nn.ModuleList`` objects.
        """

        up_decoders = nn.ModuleList()
        skip_convs = nn.ModuleList()
        for in_ch in self.dense_low_channels[1:]:
            up_decoders.append(UpBlock_attention(in_ch, in_ch // 2))
            skip_convs.append(
                nn.Sequential(
                    SELayer(in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.GELU(),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            )
        return up_decoders, skip_convs

    def _initialize_reduction_convs(self) -> nn.ModuleList:
        """Create 1x1 conv layers to reduce deep feature maps to single-channel
        masks.

        Returns
        -------
        nn.ModuleList
            ModuleList of ``nn.Conv2d(ch, 1, kernel_size=1)`` for each mask
            generation channel.
        """
        reducttion_conv = nn.ModuleList()
        for ch in self.mask_channel_gen:
            reducttion_conv.append(nn.Conv2d(ch, 1, kernel_size=1, stride=1))
        return reducttion_conv

    def _load_sam_checkpoint(self, ckpt_path):
        """Load SAM checkpoint parameters from the given path, if provided.

        The function attempts to load the checkpoint on CPU and then calls
        ``self.load_state_dict`` with ``strict=False`` to allow partial
        compatibility.
        """
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
            # sd = {k.replace("sam_mask_decoder.transformer", "decoder_transformer"): v for k, v in sd.items()}
            unexpected_keys, missing_keys = self.load_state_dict(sd, strict=False)
        print("Finish loading sam2 checkpoint")

    def _freeze_encoder(self):
        """Freeze parts of the image encoder while leaving promote generator
        and decoder transformer parameters trainable.

        This helper sets ``requires_grad`` appropriately based on parameter
        name patterns.
        """
        for name, para in self.named_parameters():
            if "image_encoder.trunk" in name and "promote_genertor" not in name:
                para.requires_grad_(False)
            if "image_encoder.neck" in name and "promote_genertor" not in name:
                para.requires_grad_(True)
            elif "decoder_transformer" in name:
                para.requires_grad_(True)

    def print_param_quantity(self):
        """Print a simple summary of parameter quantities (in millions).

        The printed value reports the total model parameter count adjusted to
        exclude the frozen encoder trunk while including the promote generator
        parameters.
        """
        trunk_param = sum(p.numel() for p in self.image_encoder.trunk.parameters()) / 1_000_000
        pmtg_param = sum(p.numel() for p in self.image_encoder.trunk.promote_genertor.parameters()) / 1_000_000
        all_param = sum(p.numel() for p in self.parameters()) / 1_000_000
        print(f"The parameter number of the model is {all_param - trunk_param + pmtg_param:.2f}M")

    def _process_deep_features(self, features: dict) -> list:
        """Process deep features with the decoder transformer and hypernetwork.

        This method:
        - Selects the deepest available image embeddings.
        - Computes multi-scale positional encodings.
        - Runs the decoder transformer producing token features and a
          transformed source representation.
        - Applies a hypernetwork MLP to obtain scaling factors which are used
          to modulate the upscaled embedding.
        - Projects and refines the result with ``proj_block`` and then uses
          upsampling decoder blocks and skip connections to create a list of
          multi-scale deep feature maps (not yet reduced to single-channel
          masks).

        Parameters
        ----------
        features : dict
            Dictionary returned by :class:`ImageEncoder` expected to contain
            keys ``"dense_embeds"`` and ``"sam_backbone_embeds"``.

        Returns
        -------
        list
            List of multi-scale deep features.
        """
        masks = []
        dense_features, sam_feature = features["dense_embeds"], features["sam_backbone_embeds"]
        try:
            image_embeddings = sam_feature[-1]
        except IndexError:
            image_embeddings = dense_features[-1]

        pe_input = dense_features + sam_feature
        pe_input = pe_input[:len(self.pe_inch)]
        image_pe = self.image_pe_encoder(pe_input)

        B, C, W, H = image_embeddings.shape
        src = self.deep_conv_block(image_embeddings)
        token = self.query_generator(src)
        hs, src = self.decoder_transformer(src, image_pe, token)
        src = src.transpose(1, 2).contiguous().view(B, self.decoder_dim, W, H)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in = self.output_hypernetworks_mlp(hs.squeeze(1))
        deep_feat = hyper_in.unsqueeze(-1).unsqueeze(-1) * upscaled_embedding

        deep_feat = self.proj_block(deep_feat)
        for i, (feature_map, skip_conv, up_decoder) in enumerate(zip(dense_features[::-1], self.skip_convs, self.up_decoders)):
            deep_feat = up_decoder(deep_feat, skip_conv(feature_map))
            masks.append(deep_feat)

        return masks

    def _process_deep_features2(self, features: dict) -> list:
        """Alternative processing path used when SAM decoder is not enabled.

        This simpler path concatenates dense and SAM features, reverses the
        order and runs projection + upsampling modules to produce multi-scale
        deep features.
        """
        masks = []
        dense_features, sam_feature = features["dense_embeds"], features["sam_backbone_embeds"]

        dense_features = (dense_features + sam_feature)[::-1]

        deep_feat = self.proj_block(dense_features[0])

        for i, (feature_map, skip_conv, up_decoder) in enumerate(zip(dense_features[1:], self.skip_convs, self.up_decoders)):
            deep_feat = up_decoder(deep_feat, skip_conv(feature_map))
            masks.append(deep_feat)

        return masks[-len(self.mask_channel_gen):]

    def _generate_masks(self, deep_feats: list, image_size: list[int, int]) -> list:
        """Reduce multi-scale deep features to single-channel masks and resize.

        For each deep feature map, this method resizes it to ``image_size``
        using bilinear interpolation and applies a 1x1 conv to produce the
        final single-channel mask tensor.
        """
        masks = []
        for mask_conv, feature_map in zip(self.reduction_convs[::-1], deep_feats[::-1]):
            mask_0 = F.interpolate(feature_map, image_size, mode="bilinear", align_corners=False)
            mask_0 = mask_conv(mask_0)
            masks.append(mask_0)

        return masks

    def forward(self, x: torch.tensor, warm_flag):
        """Forward method: produce multi-scale masks from input images.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        list[torch.Tensor]
            List of mask tensors, each shaped (B, 1, H, W).
        """
        out_image_size = x.shape[-2:]
        features = self.image_encoder(x)
        # The x_py_init module expects an image input (x)
        x_py_init = self.hdnet_init_conv(x)
        if self.use_sam_decoder:
            masks = self._process_deep_features(features)
        else:
            masks = self._process_deep_features2(features)

        if warm_flag == True:

            masks = self._generate_masks(masks, out_image_size)
            mask0 = masks[0]  # The first mask tensor
            mask1 = masks[1]  # The second mask tensor
            mask2 = masks[2]  # The third mask tensor
  
            
            x_py_v2 = x_py_init * self.sigmoid(mask2) + x_py_init
            x_py_v2 = self.py2(x_py_v2)
            x_py_v1 = x_py_v2 * self.sigmoid(mask1) + x_py_v2
            x_py_v1 = self.py1(x_py_v1)
            x_py_v0 = x_py_v1 * self.sigmoid(mask0) + x_py_v1
            x_py_v0 = self.sigmoid(self.py0(x_py_v0))
            output = self.final(torch.cat([mask0, mask1, mask2], dim=1))
            output = output * x_py_v0 + output

            masks[0] = output
    
        
        else:
 
            masks = self._generate_masks(masks, out_image_size)
        
 
        return masks


def make_adaptor(
    backbone_channel_list: list[int] = [384, 192, 96],
    dense_low_channels: list[int] = [96, 48, 24],
    stages: list[int] = [1, 2, 7],
    global_att_blocks: list[int] = [5, 7, 9],
    window_pos_embed_bkg_spatial_size: list[int] = [7, 7],
    window_spec: list[int] = [8, 4, 16],
    block: str = "res",
    embed_dim=96,
    use_sam_decoder=True,
    pe_inch=[24, 48, 96],
    sam_ckpt_path=None,
):
    """_summary_

    Args:
        backbone_channel_list (list[int], optional): The list of encoder channels. Defaults to [384, 192, 96].
        out_dim (int, optional): Number of masks in the final output. Defaults to 4.
        down_times (int, optional): Times of feature map dimensionality drop for shallow feature extraction. Defaults to 3.
        stages (list[int], optional): The stages of hieradet. Defaults to [1, 2, 7].
        global_att_blocks (list[int], optional): global attention blocks. Defaults to [5, 7, 9].
        window_pos_embed_bkg_spatial_size (list[int], optional): window size. Defaults to [7, 7].
        window_spec (list[int], optional): window spec. Defaults to [8, 4, 16].
        block (str, optional): The type of block used in encoder. Defaults to "res".

    Returns:
        nn.Module: sam adaptor
    """
    promote_generator = MultiScaleBlock(stages=stages, embed_dim=embed_dim)

    sam_encoder = Hiera(
        promote_genertor=promote_generator,
        embed_dim=embed_dim,
        num_heads=1,
        stages=stages,
        global_att_blocks=global_att_blocks,
        window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
        window_spec=window_spec,
    )

    decoder_transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    )

    predictor = SamAdaptor(
        sam_encoder=sam_encoder,
        decoder_transformer=decoder_transformer,
        backbone_channel_list=backbone_channel_list,
        stages=stages,
        block=block,
        dense_low_channels=dense_low_channels,
        use_sam_decoder=use_sam_decoder,
        pe_inch=pe_inch,
    )
    if sam_ckpt_path is not None:
        predictor._load_sam_checkpoint(sam_ckpt_path)
    return predictor
