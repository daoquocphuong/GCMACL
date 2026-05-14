import torch
import torch.nn as nn

class SentimentFeatureRefinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), 
            nn.ReLU(),
            nn.Linear(dim, dim), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x_p = x.transpose(1, 2)
        avg_p = self.avg_pool(x_p).squeeze(-1)
        max_p = self.max_pool(x_p).squeeze(-1)
        combined = torch.cat([avg_p, max_p], dim=-1)
        weight = self.mlp(combined)
        return (max_p * weight)
