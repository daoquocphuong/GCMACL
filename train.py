import torch
from config import CONFIG, get_lr
from utils.engine import train_model

if __name__ == "__main__":
    # Cập nhật LR động dựa trên mode
    CONFIG['lr'] = get_lr(CONFIG['train_mode'])
    
    # Thực thi training
    train_model(CONFIG)