import os
import pickle
import warnings
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging

from configs.config import ExperimentConfig
from data.dataset import MultiModalDataset
from models.multimodal_model import GCMACLFramework
from train import Trainer
from utils.metrics import print_evaluation_summary
from utils.seed import set_seed, seed_worker, get_generator

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data & Prepare Labels
    df = pd.read_csv(cfg.df_path).dropna()
    unique_labels = sorted(df["sentiment"].unique())
    label_map = {str(lbl).lower(): i for i, lbl in enumerate(unique_labels)}
    cfg.num_classes = len(unique_labels)

    # 2. Load Knowledge Dictionary
    concept_dict = {}
    if cfg.use_conceptnet and os.path.exists(cfg.conceptnet_path):
        with open(cfg.conceptnet_path, "rb") as f:
            concept_dict = pickle.load(f)

    # 3. Splits & DataLoaders
    tr_df, tmp = train_test_split(df, test_size=0.2, stratify=df["sentiment"], random_state=cfg.seed)
    val_df, ts_df = train_test_split(tmp, test_size=0.5, stratify=tmp["sentiment"], random_state=cfg.seed)

    tk = AutoTokenizer.from_pretrained(cfg.text_model, use_fast=True)
    g = get_generator(cfg.seed)

    loaders = {
        "train": DataLoader(
            MultiModalDataset(tr_df, tk, cfg, label_map, concept_dict, is_train=True),
            batch_size=cfg.bs,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            MultiModalDataset(val_df, tk, cfg, label_map, concept_dict, is_train=False),
            batch_size=cfg.bs,
            shuffle=False,
        ),
        "test": DataLoader(
            MultiModalDataset(ts_df, tk, cfg, label_map, concept_dict, is_train=False),
            batch_size=cfg.bs,
            shuffle=False,
        ),
    }

    # 4. Initialize Architecture
    model = GCMACLFramework(cfg).to(device)

    # Force LayerNorm to float32 for mixed-precision stability
    for name, module in model.named_modules():
        if "LayerNorm" in name or "layer_norm" in name:
            module.float()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print(f"[Model Complexity] Trainable: {trainable_params:,} | Frozen: {total_params - trainable_params:,} | Total: {total_params:,}")
    print("=" * 80)

    # 5. Execution
    trainer = Trainer(model, loaders, cfg, device)
    trainer.fit()

    test_preds, test_truths = trainer.test()
    print_evaluation_summary(test_truths, test_preds, target_names=unique_labels)

if __name__ == "__main__":
    main()
