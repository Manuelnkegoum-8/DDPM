import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    #assert len(timesteps.shape) == 1
    timesteps = timesteps.squeeze()
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

class WideResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels,num_groups,time_channels,dropout=0.):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.norm2 = nn.GroupNorm(num_groups,out_channels)
        self.norm = nn.GroupNorm(num_groups,in_channels)
        self.temb_proj = torch.nn.Linear(time_channels,out_channels)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, inputs,temb):
        out = self.conv1(F.relu(self.norm(inputs)))
        out = out + self.temb_proj(F.relu(temb))[:,:,None,None]
        out = F.relu(self.norm2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out+self.nin_shortcut(inputs)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

