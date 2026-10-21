"""Metric evaluation functions for Phase 1 FL and Phase 2 statistical bootstrap."""

from typing import Dict, Union
import numpy as np
import torch
import torch.nn.functional as F


def compute_accuracy(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute standard classification accuracy."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_error_rate(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute classification error rate (1.0 - accuracy)."""
    return 1.0 - compute_accuracy(y_true, y_pred)


def compute_metrics(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    err = 1.0 - acc

    # Macro F1 calculation
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_list = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "accuracy": acc,
        "error_rate": err,
        "f1_score": macro_f1,
        "sample_count": len(y_true),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluate PyTorch model on a DataLoader and return metrics and loss."""
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            elif isinstance(batch, dict):
                x, y = batch["input"].to(device), batch["target"].to(device)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            out = model(x)
            loss = F.cross_entropy(out, y)
            total_loss += loss.item() * len(y)
            total_samples += len(y)

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = float(avg_loss)
    return metrics
