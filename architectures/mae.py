import torch 
import torch.nn as nn

class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder for Vision Transformer with ViT backbone
    """
    def __init__(self, img_size = 224, patch_size = 16, channels = 3, embed_dim = 1024, depth =24, num_heads = 16, mlp_ratio = 4.,
                 decoder_embed_dim = 512, decoder_depth = 8, decoder_num_heads = 16, norm_layer = nn.LayerNorm):
        super().__init__()

        