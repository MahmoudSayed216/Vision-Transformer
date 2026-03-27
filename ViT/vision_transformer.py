import torch.nn as nn
import torch
from typing import Literal
from ViT.positional_encoding import PositionalEncoder



class PatchEncoder(nn.Module):
    def __init__(self, in_features, patch_side_length, hidden_dim, n_patches):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=hidden_dim, kernel_size=patch_side_length, stride=patch_side_length)
        self.patch_side_length = patch_side_length
        self.hiddem_dim = hidden_dim
        self.n_patches = n_patches
    def forward(self, x):
        B, C, W, H = x.shape
        x = self.conv(x)
        
        
        x = x.reshape(B, self.hiddem_dim, self.n_patches)

        x = x.permute(0, 2, 1)
        return x


class VisionTransformer(nn.Module):

    def __init__(self, size:Literal["s", "m", "l"]):
        super().__init__()
        
        self.hidden_dim = {"s": 768, "m":1024, "l":1280}[size]
        self.n_layers = {"s": 12, "m":24, "l":32}[size]
        self.mlp_head_dim = {"s": 3072, "m":4096, "l":5120}[size]
        self.n_heads = {"s": 12, "m":16, "l":16}[size]
        self.patch_dim = 16
        self.n_patches = int((224**2)/(16**2))+1 # for the [CLS] token

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.patch_encoder = nn.Linear(in_features=self.patch_dim**2, out_features=self.hidden_dim)
        self.patch_encoder = PatchEncoder(3, self.patch_dim, self.hidden_dim, n_patches=self.n_patches-1)
        self.pos_encoder = PositionalEncoder(embed_dim=self.hidden_dim, n_patches=self.n_patches)

        self.transformer_encoder = ViTEncoder(n_layers=self.n_layers, n_heads=self.n_heads, hidden_dim=self.hidden_dim, mlp_head_dim = self.mlp_head_dim)
        self.norm = nn.LayerNorm(normalized_shape=self.hidden_dim, eps=1e-6)
        self.classification_head = nn.Linear(in_features=self.hidden_dim, out_features=1000)

        ##TODO: add a patch of zeros of [CLS] token

    def forward(self, x):
        x = self.patch_encoder(x)

        batch_size = x.shape[0]
        x = torch.cat([self.class_token.expand(batch_size, -1, -1), x], dim=1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.classification_head(x)
        return x
        #? probably needs a softmax ? 


    def load_pretrained_weights(self):
        pass


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_head_dim, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-6)
        self.mlp1 = nn.Linear(in_features=hidden_dim, out_features=mlp_head_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.mlp2 = nn.Linear(in_features=mlp_head_dim, out_features=hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        o = x
        o = self.norm(o)
        o = self.mlp1(o)
        o = self.activation(o)
        o = self.dropout1(o)
        o = self.mlp2(o)
        o = self.dropout2(o)

        return o+x


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, attention_dropout, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        o = self.norm1(x)
        o, _ = self.attention(o, o, o, need_weights=False)
        o = self.dropout(o)
        x = x + o
        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, attention_dropout, mlp_head_dim):
        super().__init__()
        # self.mlp = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.attention = SelfAttentionBlock(hidden_dim=hidden_dim, n_heads=n_heads, attention_dropout=attention_dropout, dropout=dropout)
        self.mlp_block = MLPBlock(hidden_dim, mlp_head_dim, dropout=dropout)
        
    def forward(self, x):
        o = self.attention(x)
        o = self.mlp_block(o)
        return o


class ViTEncoder(nn.Module):

    def __init__(self, n_layers, n_heads, hidden_dim, mlp_head_dim):
        super().__init__()

        self.dropout = 0.0
        self.attention_dropout = 0.0


        self.blocks = nn.Sequential(*[EncoderBlock(hidden_dim=hidden_dim, n_heads=n_heads, dropout=self.dropout, attention_dropout=self.attention_dropout, mlp_head_dim=mlp_head_dim) for _ in range(n_layers)])


    def forward(self, x):

        return self.blocks(x)
    

