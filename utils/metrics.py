from typing import Dict, List
import numpy as np
from sklearn.metrics import classification_report, f1_score

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    return {
        "acc": float((arr_pred == arr_true).mean()),
        "f1_weighted": float(f1_score(arr_true, arr_pred, average="weighted")),
        "f1_macro": float(f1_score(arr_true, arr_pred, average="macro")),
    }

def print_evaluation_summary(y_true: List[int], y_pred: List[int], target_names: List[str]) -> None:
    metrics = compute_metrics(y_true, y_pred)
    print("=" * 80)
    print(
        f"Test Accuracy: {metrics['acc'] * 100:.2f}% | "
        f"Test F1-W: {metrics['f1_weighted'] * 100:.2f}% | "
        f"Test F1-M: {metrics['f1_macro'] * 100:.2f}%"
    )
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print("=" * 80)