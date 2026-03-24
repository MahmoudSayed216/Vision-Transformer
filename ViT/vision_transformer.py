import torch.nn as nn
import torch
from typing import Literal
from positional_encoding import PositionalEncoder






class VisionTransformer(nn.Module):

    def __init__(self, size:Literal["s", "m", "l"]):
        super().__init__()
        self.embedding_dim = {"s": 512, "m":768, "l":768}[size]
        self.patch_dim = 16

        self.pos_encoder = PositionalEncoder()


