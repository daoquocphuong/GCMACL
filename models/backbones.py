import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class BackboneManager(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Text Backbone
        self.text_model = AutoModel.from_pretrained(cfg['text_model'])
        # Vision Backbone
        self.img_model = AutoModel.from_pretrained(cfg['image_model'])
        
        self._apply_training_strategy()

    def _apply_training_strategy(self):
        mode = self.cfg.get('train_mode', 'full')
        
        if mode == 'frozen':
            for p in self.text_model.parameters(): p.requires_grad = False
            for p in self.img_model.parameters(): p.requires_grad = False
        
        elif mode == 'lora':
            # Chỉ giữ lại logic LoRA tinh gọn
            lora_config_text = LoraConfig(
                r=self.cfg['lora_r'], lora_alpha=self.cfg['lora_alpha'],
                target_modules=["query_proj", "value_proj"], 
                lora_dropout=0.1, bias="none"
            )
            self.text_model = get_peft_model(self.text_model, lora_config_text)
            
            lora_config_img = LoraConfig(
                r=self.cfg['lora_r'], lora_alpha=self.cfg['lora_alpha'],
                target_modules=["query", "value"],
                lora_dropout=0.1, bias="none"
            )
            self.img_model = get_peft_model(self.img_model, lora_config_img)

    def forward_text(self, ids, msk):
        return self.text_model(ids, attention_mask=msk).last_hidden_state

    def forward_img(self, imgs):
        return self.img_model(imgs).last_hidden_state
