import pytest
import torch
from dinov3.models.vision_transformer import DinoVisionTransformer

from cpdino_lab.models import adapt_encoder


@pytest.fixture
def small_model():
    torch.set_num_threads(2)
    encoder = DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=48,
        depth=2, num_heads=3, n_storage_tokens=4, layerscale_init=1e-5,
        pos_embed_rope_dtype='fp32')
    encoder.init_weights()
    return adapt_encoder(encoder, name='test-tiny', rdrop=0)
