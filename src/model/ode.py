import torch
import torch.nn as nn

from dataclasses import dataclass
from .module import SinusoidalPosEmb
from .config import ModelConfig
from ..utils import exists


@dataclass
class RNAVeloNetConfig(ModelConfig):
    # hidden dimension
    dim: int

    # input dimension
    input_dim: int

    # position embedding theta
    sinusoidal_pos_emb_theta: int = 10000

    # encoder/decoder extra hidden layer number
    hidden_layer_num: int = 0


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, batch_norm: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        self.norm = nn.BatchNorm1d(dim_out) if batch_norm else nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class FFNBlock(nn.Module):
    """https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L306"""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: int = None,
        batch_norm: bool = False
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, batch_norm=batch_norm)
        self.res_map = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        return h + self.res_map(x)


class RNAVeloNet(nn.Module):
    def __init__(self, config: RNAVeloNetConfig) -> None:
        super().__init__()
        dim = config.dim
        time_dim = dim // 2
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=config.sinusoidal_pos_emb_theta)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_proj = nn.Linear(config.input_dim, config.dim)
        self.encoder = nn.ModuleList(
            [
                FFNBlock(config.dim, 64, time_emb_dim=time_dim),
            ]
            + [
                FFNBlock(64, 64, time_emb_dim=time_dim)
                for _ in range(config.hidden_layer_num)
            ]
            + [
                FFNBlock(64, 16, time_emb_dim=time_dim),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                FFNBlock(16, 64, time_emb_dim=time_dim),
            ]
            + [
                FFNBlock(64, 64, time_emb_dim=time_dim)
                for _ in range(config.hidden_layer_num)
            ]
            + [
                FFNBlock(64, config.dim, time_emb_dim=time_dim),
            ]
        )
        self.final_proj = nn.Linear(config.dim, config.input_dim)

    def forward(self, t, x, **kwargs):
        batch_size = x.size(0)
        time = self.time_mlp(t.expand(batch_size))
        x = self.init_proj(x)

        for block in self.encoder:
            x = block(x, time_emb=time)
        for block in self.decoder:
            x = block(x, time_emb=time)

        return self.final_proj(x)


# class RNAVeloNet(nn.Module):
#     def __init__(self, config: RNAVeloNetConfig) -> None:
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(config.input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.ReLU(),
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(16, 64),
#             nn.ReLU(),
#             nn.Linear(64, config.input_dim),
#             nn.ReLU(),
#         )

#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.encoder(x)
#         return h

#     def decode(self, h: torch.Tensor) -> torch.Tensor:
#         h = self.decoder(h)
#         return h

#     def forward(self, t, x, **kwargs):
#         h = self.encode(x)
#         dx = self.decode(h)
#         return dx
