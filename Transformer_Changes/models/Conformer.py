import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Optional

# ============================================================
# EEG Conformer adapted to Medformer/TimesNet-style inputs
#
# Input from framework:
#   x_enc: (B, L, C)  where L=seq_len (time samples), C=enc_in (EEG channels)
#
# Conformer internal expects:
#   (B, 1, C, L)
#
# NOTE (important for SWA):
# - This version REMOVES LazyLinear so SWA (AveragedModel) won't crash
#   with "uninitialized parameter".
# - We compute the flatten dimension from configs.seq_len + patch embedding
#   pooling parameters (same as the original Conformer).
# ============================================================


def _conv1d_out_len(L: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    # PyTorch conv length formula
    return math.floor((L + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def _pool1d_out_len(L: int, kernel: int, stride: int) -> int:
    return math.floor((L - kernel) / stride + 1)


def _num_tokens_from_seq_len(seq_len: int) -> int:
    # Conformer shallownet:
    # Conv2d temporal kernel (1,25) stride (1,1) => width reduces: L -> L-25+1
    w1 = _conv1d_out_len(seq_len, kernel=25, stride=1, padding=0, dilation=1)
    # AvgPool2d kernel (1,75) stride (1,15)
    w2 = _pool1d_out_len(w1, kernel=75, stride=15)
    return max(1, w2)


class PatchEmbedding(nn.Module):
    """
    Convolution module from the reference Conformer implementation:
      temporal conv: (1, 25)
      spatial  conv: (C, 1) where C is EEG channels
      avg pool:      (1, 75) stride (1, 15)
      dropout
      projection: 1x1 conv to emb_size, then reshape to tokens
    """

    def __init__(self, in_ch: int, emb_size: int = 40, drop_p: float = 0.5):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1), bias=True),
            nn.Conv2d(40, 40, (in_ch, 1), (1, 1), bias=True),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(drop_p),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1), bias=True),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, 1, C, L)
        x = self.shallownet(x)     # (B, 40, 1, W)
        x = self.projection(x)     # (B, N_tokens=W, emb_size)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.last_attn = None
        self.save_attn = False
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, E)
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)


        energy = torch.einsum("bhqd,bhkd->bhqk", q, k)  # (B,H,N,N)

        if mask is not None:
            fill_value = torch.finfo(energy.dtype).min
            energy = energy.masked_fill(~mask, fill_value)

        scale = (self.emb_size ** 0.5)
        attn = F.softmax(energy / scale, dim=-1)
        self.last_attn = attn.detach().float().cpu() 
        attn = self.att_drop(attn)

        self.last_attn = attn.detach()

        if self.save_attn:
            self.last_attn = attn.detach()

        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, num_heads: int, drop_p: float):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads=num_heads, drop_p=drop_p) for _ in range(depth)])


class ClassificationHead(nn.Module):
    """
    Original Conformer FC classifier (same hidden sizes / activations),
    but with computed input dimension instead of hard-coded 2440.
    """
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N_tokens, emb_size)
        x = tokens.contiguous().view(tokens.size(0), -1)
        return self.fc(x)


class EEGConformer(nn.Module):
    def __init__(self, in_ch: int, seq_len: int, emb_size: int, depth: int, n_heads: int, dropout: float, num_class: int):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_ch=in_ch, emb_size=emb_size, drop_p=dropout)
        self.encoder = TransformerEncoder(depth=depth, emb_size=emb_size, num_heads=n_heads, drop_p=dropout)

        n_tokens = _num_tokens_from_seq_len(seq_len)
        flat_dim = n_tokens * emb_size
        self.head = ClassificationHead(in_dim=flat_dim, n_classes=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)   # (B, N, emb)
        tokens = self.encoder(tokens)  # (B, N, emb)
        logits = self.head(tokens)     # (B, num_class)
        return logits


# ============================================================
# Framework wrapper: MUST be named Model; forward signature matches
# the vanilla Transformer class in your repo.
# ============================================================

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name

        in_ch = int(configs.enc_in)
        seq_len = int(getattr(configs, "seq_len", 256))

        emb_size = int(getattr(configs, "d_model", 40))
        depth = int(getattr(configs, "e_layers", 6))
        n_heads = int(getattr(configs, "n_heads", 10))
        dropout = float(getattr(configs, "dropout", 0.5))
        num_class = int(getattr(configs, "num_class", getattr(configs, "c_out", 2)))

        # Safety: seq_len must be large enough for conv/pool chain
        if seq_len < 100:
            raise ValueError(f"Conformer expects seq_len large enough for conv/pool; got seq_len={seq_len}")

        self.net = EEGConformer(
            in_ch=in_ch,
            seq_len=seq_len,
            emb_size=emb_size,
            depth=depth,
            n_heads=n_heads,
            dropout=dropout,
            num_class=num_class,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: (B, L, C)
        if x_mark_enc is not None:
            x_enc = x_enc * x_mark_enc.unsqueeze(-1)

        # (B,L,C) -> (B,1,C,L)
        x = x_enc.permute(0, 2, 1).unsqueeze(1).contiguous()
        return self.net(x)
