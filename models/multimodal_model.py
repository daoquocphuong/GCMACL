from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.adapters import apply_train_mode
from models.layers import EmotionFeatureRefinement, IterativeCrossModalGo

class GCMACLFramework(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg

        # 1. Backbones & Projections
        self.text_backbone = AutoModel.from_pretrained(cfg.text_model)
        self.text_proj = nn.Linear(self.text_backbone.config.hidden_size, cfg.shared_dim)

        self.img_backbone = AutoModel.from_pretrained(cfg.image_model)
        self.v_proj = nn.Linear(self.img_backbone.config.hidden_size, cfg.shared_dim)

        apply_train_mode(self, cfg)

        # 2. External Knowledge & Region Embeddings
        if getattr(cfg, "use_conceptnet", False):
            self.concept_proj = nn.Linear(cfg.concept_dim, cfg.shared_dim)
            self.concept_gate = nn.Linear(cfg.shared_dim * 2, 1)

        self.reg_proj = nn.Linear(1024, cfg.shared_dim)

        # 3. Multimodal Fusion
        if cfg.use_cross_module:
            self.cross_modal = IterativeCrossModalGo(dim=cfg.shared_dim, num_layers=cfg.cross_layers)

        # 4. Feature Refinement & Contrastive Projector
        self.entxfe = EmotionFeatureRefinement(cfg.shared_dim) if cfg.use_entxfe else None
        self.envsfe = EmotionFeatureRefinement(cfg.shared_dim) if cfg.use_envsfe else None

        self.cl_projector = nn.Sequential(
            nn.Linear(cfg.shared_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.t_weight = nn.Parameter(torch.ones(1))
        self.i_weight = nn.Parameter(torch.ones(1))

        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.shared_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(
        self,
        ids: torch.Tensor,
        msk: torch.Tensor,
        imgs: torch.Tensor,
        regions: torch.Tensor,
        concepts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_out = self.text_backbone(ids, attention_mask=msk).last_hidden_state
        t_feat = self.text_proj(t_out)

        if getattr(self.cfg, "use_conceptnet", False) and concepts is not None:
            c_feat = self.concept_proj(concepts)
            gate = torch.sigmoid(self.concept_gate(torch.cat([t_feat, c_feat], dim=-1)))
            t_feat = gate * t_feat + (1.0 - gate) * c_feat

        v_out = self.img_backbone(imgs).last_hidden_state
        v_feat = self.v_proj(v_out)

        if getattr(self.cfg, "use_mvif", True):
            v_reg = self.reg_proj(regions)
            v_feat = torch.cat([v_feat, v_reg], dim=1)

        if self.cfg.use_cross_module:
            t_feat, v_feat = self.cross_modal(t_feat, v_feat)

        f_t = self.entxfe(t_feat) if self.entxfe else t_feat[:, 0, :]
        f_i = self.envsfe(v_feat) if self.envsfe else torch.max(v_feat, dim=1)[0]

        fusion_feat = torch.cat([f_t, f_i], dim=-1)

        cl_loss = torch.tensor(0.0, device=ids.device)
        if self.training and self.cfg.use_cl:
            z_t = F.normalize(self.cl_projector(f_t), dim=-1)
            z_i = F.normalize(self.cl_projector(f_i), dim=-1)
            logits_per_text = torch.matmul(z_t, z_i.t()) / self.cfg.cl_temp
            labels = torch.arange(ids.size(0), device=ids.device)
            w = torch.softmax(torch.cat([self.t_weight, self.i_weight]), dim=0)
            cl_loss = w[0] * F.cross_entropy(logits_per_text, labels) + w[1] * F.cross_entropy(logits_per_text.t(), labels)

        return self.classifier(fusion_feat), cl_loss