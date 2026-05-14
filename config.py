import os

CONFIG = {
    'text_model': 'microsoft/deberta-v3-base',
    'image_model': 'microsoft/swin-small-patch4-window7-224',
    'train_mode': 'full',  # 'full', 'lora', 'frozen'
    'lora_r': 16,
    'lora_alpha': 32,
    'use_text_cnn_lstm': False,
    'use_img_cnn': False,
    'use_gate_attention': False,
    'use_conceptnet': True,
    'use_mvif': True,
    'use_entxfe': True,
    'use_envsfe': True,
    'use_cross_module': True,
    'use_pos_embed': False,
    'use_cl': True,
    'shared_dim': 768,
    'concept_dim': 300,
    'cross_layers': 5,
    'cl_weight': 0.2,
    'cl_temp': 0.05,
    'max_len': 80,
    'num_regions': 2,
    'bs': 32,
    'epochs': 10,
    'patience': 3,
    'monitor_metric': 'f1_weighted',
    'dropout': 0.2,
    'df_path': '/path/to/your/dataset.csv',
    'feature_dir': '/path/to/your/features',
    'conceptnet_path': 'conceptnet_embeddings.pkl'
}

def get_lr(mode):
    if mode == 'lora': return 2e-4
    if mode == 'frozen': return 2e-3
    return 2e-5