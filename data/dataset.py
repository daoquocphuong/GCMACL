import os
import re
from typing import Any, Dict, List, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.preprocessing import preprocess_text

class MultiModalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        cfg: Any,
        label_map: Dict[str, int],
        concept_dict: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
    ):
        self.df = df
        self.tk = tokenizer
        self.cfg = cfg
        self.label_map = label_map
        self.label_names = list(label_map.keys())
        self.concept_dict = concept_dict or {}

        t_list = [transforms.Resize((224, 224))]
        if is_train:
            t_list += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomRotation(20),
            ]
        t_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.aug = transforms.Compose(t_list)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        txt = preprocess_text(row["text_or_caption"], self.label_names)

        enc = self.tk(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        offsets = enc["offset_mapping"][0]

        c_feats = torch.zeros((self.cfg.max_len, self.cfg.concept_dim))
        if getattr(self.cfg, "use_conceptnet", False):
            special_ids = {self.tk.cls_token_id, self.tk.sep_token_id, self.tk.pad_token_id}
            for pos, (start, end) in enumerate(offsets):
                if start == end and input_ids[pos].item() in special_ids:
                    continue
                word_clean = re.sub(r"[^\w\s]", "", txt[start:end].strip().lower())
                if word_clean in self.concept_dict:
                    c_feats[pos] = torch.tensor(self.concept_dict[word_clean], dtype=torch.float)

        img_path = row["image_path"]
        try:
            img = self.aug(Image.open(img_path).convert("RGB"))
        except Exception:
            img = torch.zeros((3, 224, 224))

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        feat_path = os.path.join(self.cfg.feature_dir, f"{img_name}.pt")

        if os.path.exists(feat_path):
            reg = torch.load(feat_path, map_location="cpu")[: self.cfg.num_regions]
            if reg.size(0) < self.cfg.num_regions:
                pad_size = self.cfg.num_regions - reg.size(0)
                reg = torch.cat([reg, torch.zeros((pad_size, 1024))], dim=0)
        else:
            reg = torch.zeros((self.cfg.num_regions, 1024))

        label = self.label_map[str(row["sentiment"]).lower()]

        return {
            "ids": input_ids,
            "msk": attention_mask,
            "img": img,
            "reg": reg,
            "concepts": c_feats,
            "y": torch.tensor(label, dtype=torch.long),
        }