import torch
import torch.nn as nn


# ----------------------------
# Layout adapter (the fix)
# ----------------------------
class LayoutAdapter(nn.Module):
    """
    Adapts input layout for augmentations.

    Accepts:
      - (B, C, T)  -> used as-is
      - (B, T, D)  -> converted to (B, D, T) for augmentation, then converted back
      - (B, T, C, K) -> flattened to (B, T, C*K) -> (B, C*K, T) for augmentation, then back

    Returns tensor in the SAME layout as input.
    """
    def __init__(self, aug: nn.Module):
        super().__init__()
        self.aug = aug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        orig_dim = x.dim()

        # Handle 4D banded: (B,T,C,K) -> flatten features
        if orig_dim == 4:
            B, T, C, K = x.shape
            x2 = x.reshape(B, T, C * K)     # (B,T,D)
            # treat as (B,T,D) from here
            orig_dim = 3
            x = x2

        if orig_dim != 3:
            raise ValueError(f"Augmentation expected 3D (or 4D banded) input, got shape {tuple(x.shape)}")

        B, d1, d2 = x.shape

        # Decide whether input is (B,C,T) or (B,T,D)
        # Heuristic: time dimension is usually larger (e.g., 256) than channels/features.
        # - If second dim is large and third dim is smaller -> likely (B,T,D)
        # - Else assume (B,C,T)
        if d1 >= d2:
            # likely (B,T,D) -> convert to (B,D,T)
            x_ct = x.transpose(1, 2)  # (B,D,T)
            out = self.aug(x_ct)
            # back to (B,T,D)
            return out.transpose(1, 2)
        else:
            # likely (B,C,T)
            return self.aug(x)


# ----------------------------
# Augmentations (unchanged)
# ----------------------------
class Jitter(nn.Module):
    # apply noise on each element
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * self.scale
        return x


class Scale(nn.Module):
    # scale each channel by a random scalar
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            x = x * (1 + torch.randn(B, C, 1, device=x.device) * self.scale)
        return x


class Flip(nn.Module):
    # left-right flip
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1, device=x.device) < self.prob:
            return torch.flip(x, [-1])
        return x


class Shuffle(nn.Module):
    # shuffle channels order
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1, device=x.device) < self.prob:
            B, C, T = x.shape
            perm = torch.randperm(C, device=x.device)
            return x[:, perm, :]
        return x


class TemporalMask(nn.Module):
    # Randomly mask a portion of timestamps across all channels
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(T * self.ratio)
            if num_mask > 0:
                mask_indices = torch.randperm(T, device=x.device)[:num_mask]
                x = x.clone()
                x[:, :, mask_indices] = 0
        return x


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            x_fft = torch.fft.rfft(x, dim=-1)
            mask = (torch.rand(x_fft.shape, device=x.device) > self.ratio)
            x_fft = x_fft * mask
            x = torch.fft.irfft(x_fft, n=T, dim=-1)
        return x


def get_augmentation(augmentation: str) -> nn.Module:
    """
    Returns an augmentation wrapped with LayoutAdapter so it can accept:
      (B,C,T) OR (B,T,D) OR (B,T,C,K)
    """
    if augmentation.startswith("jitter"):
        aug = Jitter() if len(augmentation) == 6 else Jitter(float(augmentation[6:]))
    elif augmentation.startswith("scale"):
        aug = Scale() if len(augmentation) == 5 else Scale(float(augmentation[5:]))
    elif augmentation.startswith("drop"):
        aug = nn.Dropout(0.1) if len(augmentation) == 4 else nn.Dropout(float(augmentation[4:]))
    elif augmentation.startswith("flip"):
        aug = Flip() if len(augmentation) == 4 else Flip(float(augmentation[4:]))
    elif augmentation.startswith("shuffle"):
        aug = Shuffle() if len(augmentation) == 7 else Shuffle(float(augmentation[7:]))
    elif augmentation.startswith("frequency"):
        aug = FrequencyMask() if len(augmentation) == 9 else FrequencyMask(float(augmentation[9:]))
    elif augmentation.startswith("mask"):
        aug = TemporalMask() if len(augmentation) == 4 else TemporalMask(float(augmentation[4:]))
    elif augmentation == "none":
        aug = nn.Identity()
    else:
        raise ValueError(f"Unknown augmentation {augmentation}")

    # Wrap everything (including Identity is fine)
    return LayoutAdapter(aug)