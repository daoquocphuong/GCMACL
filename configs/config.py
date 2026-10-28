from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class ExperimentConfig:
    # Reproducibility & Device
    seed: int = 42

    # Model Backbones
    text_model: str = "microsoft/deberta-v3-base"
    image_model: str = "microsoft/swin-tiny-patch4-window7-224"
    train_mode: str = "full"  # 'full', 'frozen', 'lora'
    lora_r: int = 16
    lora_alpha: int = 32

    # Architecture Flags & Dims
    use_conceptnet: bool = True
    use_mvif: bool = True
    use_entxfe: bool = True
    use_envsfe: bool = True
    use_cross_module: bool = True
    use_cl: bool = True

    shared_dim: int = 768
    concept_dim: int = 300
    cross_layers: int = 5
    cl_weight: float = 0.2
    cl_temp: float = 0.05
    max_len: int = 64
    num_regions: int = 8
    dropout: float = 0.2
    num_classes: Optional[int] = None

    # Training Hyperparameters
    lr: float = 2e-5
    bs: int = 16
    wd: float = 0.02
    epochs: int = 10
    patience: int = 3
    monitor_metric: str = "f1_weighted"

    # File Paths
    df_path: str = "/home/quocphuong/MEA/Datasets/MVSA/MVSA-single/mvsa_single.csv"
    feature_dir: str = "/home/quocphuong/MEA/Datasets/MVSA/MVSA-single/features_multiobj"
    conceptnet_path: str = "conceptnet_embeddings.pkl"
    checkpoint_dir: str = "checkpoints"
