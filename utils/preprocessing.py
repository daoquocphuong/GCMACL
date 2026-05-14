import re, emoji, random, torch
import torch.nn as nn

def preprocess_text(text, label_names):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    for label in label_names:
        text = re.sub(fr"#\b{re.escape(str(label).lower())}\b", "", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    return text.strip()
