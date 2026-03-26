import torch.nn as nn
import torch
from typing import Literal
from positional_encoding import PositionalEncoder






class VisionTransformer(nn.Module):

    def __init__(self, size:Literal["s", "m", "l"]):
        super().__init__()
        self.embedding_dim = {"s": 512, "m":768, "l":768}[size]
        self.patch_dim = 16
        self.n_pathces = (224**2)/(16**2)+1 # for the [CLS] token
        
        self.patch_encoder = nn.Linear(in_features=self.patch_dim**2, out_features=self.embedding_dim)
        self.pos_encoder = PositionalEncoder(embed_dim=self.embedding_dim, n_patches=self.n_pathces+1)


        self.transformer_encoder = ViTEncoder()
        self.classification_head = nn.Linear(in_features=self.embedding_dim, out_features=1000)



    def forward(self, x):
        pass



class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass



class ViTEncoder(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass