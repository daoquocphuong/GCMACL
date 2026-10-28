from typing import Any, Tuple
import torch
import torch.nn as nn

class EmotionFeatureRefinement(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L]
        x_p = x.transpose(1, 2)
        avg_p = self.avg_pool(x_p).squeeze(-1)
        max_p = self.max_pool(x_p).squeeze(-1)

        combined = torch.cat([avg_p, max_p], dim=-1)
        weight = self.mlp(combined)
        return max_p * weight


class IterativeCrossModalGo(nn.Module):
    def __init__(self, dim: int, num_layers: int, nhead: int = 8):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "t2i_cross": nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=0.1),
                "i2t_cross": nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=0.1),
                "ln_t": nn.LayerNorm(dim),
                "ln_i": nn.LayerNorm(dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, t: torch.Tensor, i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            global_ctx = torch.cat([t, i], dim=1)

            t_res, t_weights = layer["t2i_cross"](query=t, key=global_ctx, value=global_ctx)
            t_gate = torch.sigmoid(t_weights.max(dim=-1, keepdim=True)[0] / self.temperature)
            t = layer["ln_t"](t + t_res * t_gate)

            i_res, i_weights = layer["i2t_cross"](query=i, key=global_ctx, value=global_ctx)
            i_gate = torch.sigmoid(i_weights.max(dim=-1, keepdim=True)[0] / self.temperature)
            i = layer["ln_i"](i + i_res * i_gate)

        return t, i