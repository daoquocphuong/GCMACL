import torch.nn as nn
import torch.nn.functional as F
from .backbones import BackboneManager
from .fusion import CrossModalAttention, EmotionRefinement

class GCMACLFramework(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = BackboneManager(cfg)
        d = cfg['shared_dim']
        
        # Projectors
        self.text_proj = nn.Linear(self.backbone.text_model.config.hidden_size, d)
        self.v_proj = nn.Linear(self.backbone.img_model.config.hidden_size, d)
        self.reg_proj = nn.Linear(1024, d)
        
        # ConceptNet integration
        if cfg.get('use_conceptnet'):
            self.c_proj = nn.Linear(cfg['concept_dim'], d)
            self.c_gate = nn.Linear(d * 2, 1)

        self.cross_modal = CrossModalAttention(d, layers=cfg['cross_layers'])
        self.refine_t = EmotionRefinement(d)
        self.refine_i = EmotionRefinement(d)
        
        # Contrastive head
        self.cl_proj = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128))
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(512, cfg['num_classes'])
        )

    def forward(self, ids, msk, imgs, regions, concepts=None):
        # 1. Feature Extraction
        t_feat = self.text_proj(self.backbone.forward_text(ids, msk))
        v_feat = self.v_proj(self.backbone.forward_img(imgs))
        
        # 2. Knowledge Enhancement
        if concepts is not None:
            c_f = self.c_proj(concepts)
            g = torch.sigmoid(self.c_gate(torch.cat([t_feat, c_f], dim=-1)))
            t_feat = g * t_feat + (1 - g) * c_f
            
        # 3. Object-level fusion (MVIF)
        v_feat = torch.cat([v_feat, self.reg_proj(regions)], dim=1)
        
        # 4. Interaction & Pooling
        t_feat, v_feat = self.cross_modal(t_feat, v_feat)
        f_t, f_i = self.refine_t(t_feat), self.refine_i(v_feat)
        
        # 5. Contrastive Loss (Internal logic)
        cl_loss = torch.tensor(0.0).to(ids.device)
        if self.training:
            z_t, z_i = F.normalize(self.cl_proj(f_t), dim=-1), F.normalize(self.cl_proj(f_i), dim=-1)
            # Simple Symmetric Contrastive
            sim = torch.matmul(z_t, z_i.t()) / 0.05
            labels = torch.arange(ids.size(0)).to(ids.device)
            cl_loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)

        return self.classifier(torch.cat([f_t, f_i], dim=-1)), cl_loss
