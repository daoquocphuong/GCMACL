import torch
import torch.nn as nn

class IterativeCrossModalGo(nn.Module):
    def __init__(self, dim, num_layers, nhead=8, max_seq_len=64, num_patches=256, use_pos_embed=True):
        super().__init__()
        self.use_pos_embed = use_pos_embed
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        if self.use_pos_embed:
            self.text_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
            self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            nn.init.trunc_normal_(self.text_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.img_pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                't_cross': nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=0.1),
                'i_cross': nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=0.1),
                'ln_t': nn.LayerNorm(dim), 
                'ln_i': nn.LayerNorm(dim)
            }) for _ in range(num_layers)
        ])

    def forward(self, t, i):
        if self.use_pos_embed:
            t = t + self.text_pos_embed[:, :t.size(1), :]
            i = i + self.img_pos_embed[:, :i.size(1), :]

        for layer in self.layers:
            global_ctx = torch.cat([t, i], dim=1) 
            t_res, t_weights = layer['t_cross'](t, global_ctx, global_ctx)
            t_res = t_res * (t_weights.max(dim=-1, keepdim=True)[0] / self.temperature).sigmoid()
            t = layer['ln_t'](t + t_res)
            
            i_res, i_weights = layer['i_cross'](i, global_ctx, global_ctx)
            i_res = i_res * (i_weights.max(dim=-1, keepdim=True)[0] / self.temperature).sigmoid()
            i = layer['ln_i'](i + i_res)
        return t, i
