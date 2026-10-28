from typing import Any
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def apply_train_mode(model: nn.Module, cfg: Any) -> nn.Module:
    mode = getattr(cfg, "train_mode", "full")

    if mode == "frozen":
        print("[Training Setup] Mode: FROZEN -> Freezing feature backbones.")
        for param in model.text_backbone.parameters():
            param.requires_grad = False
        for param in model.img_backbone.parameters():
            param.requires_grad = False

    elif mode == "lora":
        print("[Training Setup] Mode: LoRA -> Injecting Low-Rank Adaptation.")
        text_lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model.text_backbone = get_peft_model(model.text_backbone, text_lora_config)

        img_lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        model.img_backbone = get_peft_model(model.img_backbone, img_lora_config)

    else:
        print("[Training Setup] Mode: FULL -> End-to-end full network fine-tuning.")

    return model