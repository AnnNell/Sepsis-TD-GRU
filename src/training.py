"""
Training harness.

Key features
------------
- Identical training budget across all model variants (epochs, patience, LR).
- Validation PR-AUC monitor with early stopping (matches manuscript).
- Multi-seed driver: trains the same model across N seeds and reports
  mean ± std of held-out test metrics.
- Persists per-seed test predictions for downstream bootstrap.
- Logs everything to a single results.json per model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import get_device, set_global_seed


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------
def build_loader(
    X: np.ndarray, M: np.ndarray, D: np.ndarray, y: np.ndarray,
    batch_size: int, shuffle: bool, drop_last: bool = False,
) -> DataLoader:
    """Wrap numpy tensors into a PyTorch DataLoader.

    y is (N, 2) -> [label, time_to_onset]; passed through unchanged so the
    ATP loss can read the second column.
    """
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(M.astype(np.float32)),
        torch.from_numpy(D.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


# -----------------------------------------------------------------------------
# Single training run
# -----------------------------------------------------------------------------
def train_one_run(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    seed: int,
    log_prefix: str = "",
) -> Tuple[nn.Module, Dict]:
    """Train a single model with early stopping. Returns (best_model, history)."""
    device = get_device()
    model = model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
    )

    best_val_pr = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_left = cfg["training"]["early_stopping_patience"]
    history = {"train_loss": [], "val_pr_auc": [], "val_roc_auc": []}

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        running = 0.0
        n_batches = 0
        for X, M, D, y in train_loader:
            X = X.to(device); M = M.to(device); D = D.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(X, M, D)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            opt.step()
            running += float(loss.item())
            n_batches += 1

        # Validation pass
        val_probs, val_labels = predict_loader(model, val_loader, device)
        val_pr = average_precision_score(val_labels, val_probs)
        val_roc = roc_auc_score(val_labels, val_probs)
        history["train_loss"].append(running / max(1, n_batches))
        history["val_pr_auc"].append(float(val_pr))
        history["val_roc_auc"].append(float(val_roc))

        improved = val_pr > best_val_pr + 1e-6
        if improved:
            best_val_pr = float(val_pr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["training"]["early_stopping_patience"]
        else:
            patience_left -= 1

        print(
            f"{log_prefix}seed={seed} ep={epoch+1:02d} "
            f"loss={running/max(1,n_batches):.4f} "
            f"val_pr={val_pr:.4f} val_roc={val_roc:.4f} "
            f"best_val_pr={best_val_pr:.4f} pat={patience_left}"
        )
        if patience_left <= 0:
            print(f"{log_prefix}early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    history["best_val_pr_auc"] = best_val_pr
    return model, history


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) for a loader."""
    model.eval()
    probs, labels = [], []
    for X, M, D, y in loader:
        X = X.to(device); M = M.to(device); D = D.to(device)
        logits = model(X, M, D)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(y[:, 0].numpy())
    return np.concatenate(probs), np.concatenate(labels)


# -----------------------------------------------------------------------------
# Multi-seed driver
# -----------------------------------------------------------------------------
def train_multi_seed(
    model_name: str,
    model_factory: Callable[[int], nn.Module],
    loss_factory: Callable[[int], nn.Module],
    data_arrays: Dict[str, np.ndarray],
    cfg: Dict,
    out_dir: Path,
    seeds: Optional[List[int]] = None,
) -> Dict:
    """Train a model across multiple seeds and aggregate metrics.

    Parameters
    ----------
    model_name : short identifier; used for filenames.
    model_factory : function(seed) -> nn.Module (fresh model each call).
    loss_factory  : function(seed) -> nn.Module (loss may be seed-independent).
    data_arrays   : dict with X_train, M_train, D_train, y_train, ... and test.
    out_dir       : where predictions and results.json go.

    Returns
    -------
    A dict with mean/std test metrics and per-seed predictions on test.
    """
    if seeds is None:
        seeds = cfg["seeds"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_probs_per_seed: List[np.ndarray] = []
    histories: List[Dict] = []
    test_metrics: List[Dict] = []

    for seed in seeds:
        set_global_seed(seed)
        model = model_factory(seed)
        loss_fn = loss_factory(seed)
        train_loader = build_loader(
            data_arrays["X_train"], data_arrays["M_train"],
            data_arrays["D_train"], data_arrays["y_train"],
            batch_size=cfg["training"]["batch_size"], shuffle=True,
        )
        val_loader = build_loader(
            data_arrays["X_val"], data_arrays["M_val"],
            data_arrays["D_val"], data_arrays["y_val"],
            batch_size=cfg["training"]["batch_size"], shuffle=False,
        )
        test_loader = build_loader(
            data_arrays["X_test"], data_arrays["M_test"],
            data_arrays["D_test"], data_arrays["y_test"],
            batch_size=cfg["training"]["batch_size"], shuffle=False,
        )

        model, hist = train_one_run(
            model, loss_fn, train_loader, val_loader,
            cfg, seed, log_prefix=f"[{model_name}] ",
        )
        histories.append(hist)

        test_probs, test_labels = predict_loader(model, test_loader, get_device())
        val_probs_seed, _ = predict_loader(model, val_loader, get_device())
        test_probs_per_seed.append(test_probs)
        m = {
            "seed": seed,
            "test_pr_auc": float(average_precision_score(test_labels, test_probs)),
            "test_roc_auc": float(roc_auc_score(test_labels, test_probs)),
            "test_brier": float(brier_score_loss(test_labels, test_probs)),
            "best_val_pr_auc": float(hist["best_val_pr_auc"]),
        }
        test_metrics.append(m)

        np.save(out_dir / f"{model_name}_seed{seed}_test_probs.npy", test_probs)
        np.save(out_dir / f"{model_name}_seed{seed}_val_probs.npy", val_probs_seed)

    # Aggregate
    test_pr = np.array([m["test_pr_auc"] for m in test_metrics])
    test_roc = np.array([m["test_roc_auc"] for m in test_metrics])
    test_brier = np.array([m["test_brier"] for m in test_metrics])

    summary = {
        "model": model_name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_seed": test_metrics,
        "test_pr_auc_mean": float(test_pr.mean()),
        "test_pr_auc_std": float(test_pr.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "test_roc_auc_mean": float(test_roc.mean()),
        "test_roc_auc_std": float(test_roc.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "test_brier_mean": float(test_brier.mean()),
        "test_brier_std": float(test_brier.std(ddof=1)) if len(seeds) > 1 else 0.0,
    }

    # Save mean predictions across seeds (used for downstream comparison)
    mean_probs = np.mean(np.stack(test_probs_per_seed), axis=0)
    np.save(out_dir / f"{model_name}_mean_test_probs.npy", mean_probs)

    # Also aggregate val predictions for proper isotonic calibration in Notebook 5
    val_arrays = [
        np.load(out_dir / f"{model_name}_seed{s}_val_probs.npy") for s in seeds
    ]
    mean_val = np.mean(np.stack(val_arrays), axis=0)
    np.save(out_dir / f"{model_name}_mean_val_probs.npy", mean_val)

    with open(out_dir / f"{model_name}_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"\n[{model_name}] FINAL  "
        f"PR-AUC = {summary['test_pr_auc_mean']:.4f} ± {summary['test_pr_auc_std']:.4f}  "
        f"ROC-AUC = {summary['test_roc_auc_mean']:.4f} ± {summary['test_roc_auc_std']:.4f}\n"
    )
    return summary
