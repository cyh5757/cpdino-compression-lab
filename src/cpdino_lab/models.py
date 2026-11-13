"""Inject a DINOv3 encoder into the upstream CPDINO forward path."""
import torch
from torch import nn
from cellpose.vit import BaseModel, CPDINO
from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16


class AdaptedCPDINO(CPDINO):
    """CPDINO-compatible model without modifying upstream global factories."""

    def __init__(self, encoder: nn.Module, name: str, rdrop: float = .4):
        dtype = encoder.patch_embed.proj.weight.dtype
        BaseModel.__init__(self, dtype=dtype)
        if encoder.storage_tokens.shape[1] != 4:
            raise ValueError('CPDINO forward requires four register tokens')
        if encoder.patch_embed.proj.kernel_size != (16, 16):
            raise ValueError('Expected a DINOv3 patch-16 encoder')
        if not 0 <= rdrop < 1:
            raise ValueError('rdrop must be in [0, 1)')
        self.encoder = encoder
        self.model_name = name
        self.backbone = 'dino_' + name
        self.rdrop, self.ps, self.nout = rdrop, 8, 3
        self.encoder.patch_embed.proj.stride = (8, 8)
        self.encoder.patch_embed.proj.padding = (4, 4)
        # Upstream forward uses .data here. Make that frozen behavior explicit.
        self.encoder.patch_embed.proj.requires_grad_(False)
        dim = self.encoder.patch_embed.proj.out_channels
        device = self.encoder.patch_embed.proj.weight.device
        self.out = nn.Linear(dim, 192, dtype=dtype, device=device)
        self.W2 = nn.Parameter(torch.eye(192, dtype=dtype, device=device).reshape(192, 3, 8, 8), requires_grad=False)
        self.to(device)

    def forward(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[1] not in (1, 2, 3):
            raise ValueError('Expected images [N, C, H, W] with 1–3 channels')
        if min(images.shape[-2:]) < 16 or any(size % 8 for size in images.shape[-2:]):
            raise ValueError('Height and width must be >=16 and divisible by 8')
        return super().forward(images)


def adapt_encoder(encoder: nn.Module, name: str = 'vits', rdrop: float = .4) -> AdaptedCPDINO:
    return AdaptedCPDINO(encoder, name, rdrop)


def build_model(name: str = 'vits') -> AdaptedCPDINO:
    """Construct random weights locally. Use weights.load_weights for initialization."""
    factories = {'vits': dinov3_vits16, 'vitb': dinov3_vitb16, 'vitl': dinov3_vitl16}
    if name not in factories:
        raise ValueError(f'Unsupported backbone {name!r}')
    return adapt_encoder(factories[name](pretrained=False), name)
