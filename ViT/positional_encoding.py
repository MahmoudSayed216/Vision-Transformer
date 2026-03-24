import torch.nn as nn
import torch


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, n_patches):
        super().__init__()
        
        self.pos_encoding = nn.Parameter(torch.randn(1, n_patches+1, embed_dim))


    
    def forward(self, x):
        return x + self.pos_encoding