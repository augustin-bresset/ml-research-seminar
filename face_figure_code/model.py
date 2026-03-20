import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half) / (half - 1)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t[:, None] * self.freqs
        return torch.cat([a.sin(), a.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, out_dim, bias=False),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.time_proj = nn.Linear(time_dim, out_ch, bias=False)
        self.shortcut  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x)) + self.time_proj(t_emb)[:, :, None, None]
        return self.act(self.conv2(h)) + self.shortcut(x)


class UNetFM(nn.Module):
    """Bias-free UNet for OT-Flow Matching. Predicts velocity v_θ(x_t, t)."""

    def __init__(self, img_ch: int = 1, base_ch: int = 64,
                 ch_mults: tuple = (1, 2, 4, 8), time_dim: int = 256):
        super().__init__()
        chs = [base_ch * m for m in ch_mults]
        n   = len(chs)
        self.time_emb   = TimeEmbedding(base_ch, time_dim)
        self.enc_in     = nn.Conv2d(img_ch, chs[0], 3, padding=1, bias=False)
        self.enc_blocks = nn.ModuleList([ConvBlock(chs[i], chs[i+1], time_dim) for i in range(n-1)])
        self.downs      = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n-1)])
        self.mid        = ConvBlock(chs[-1], chs[-1], time_dim)
        self.ups        = nn.ModuleList([
            nn.ConvTranspose2d(chs[n-1-i], chs[n-2-i], 2, stride=2, bias=False) for i in range(n-1)
        ])
        self.dec_blocks = nn.ModuleList([
            ConvBlock(chs[n-2-i] * 2, chs[n-2-i], time_dim) for i in range(n-1)
        ])
        self.out_conv = nn.Conv2d(chs[0], img_ch, 1, bias=False)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        h  = self.enc_in(x)
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            skips.append(h); h = block(h, te); h = down(h)
        h = self.mid(h, te)
        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h); h = torch.cat([h, skip], dim=1); h = block(h, te)
        return self.out_conv(h)
