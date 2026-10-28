import re
import emoji
from typing import List

def preprocess_text(text: str, label_names: List[str]) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    for label in label_names:
        text = re.sub(rf"#\b{re.escape(str(label).lower())}\b", "", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"\s+", " ", text)
    return text.strip()