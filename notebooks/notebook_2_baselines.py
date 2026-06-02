# %% [markdown]
# # Notebook 2 — Baseline Models
#
# Six baselines, all evaluated under **identical training budgets** and on
# the **same held-out test split**. Each neural model is trained over `cfg['seeds']`
# different seeds and we report mean ± SD test PR-AUC / ROC-AUC / Brier.

# %%
# === Colab setup (run once per notebook) ===
from google.colab import drive
drive.mount('/content/drive')

import os, sys
PROJECT_ROOT = '/content/drive/MyDrive/Clinical ML Architect: Sepsis-6H Pipeline/sepsis_pipeline_6h'
os.chdir(PROJECT_ROOT)               # so relative paths like configs/config.yaml work
sys.path.insert(0, PROJECT_ROOT)     # so `from src.xyz import ...` works

# Install any missing deps
!pip install -q pyyaml xgboost
# torch, numpy, pandas, sklearn, matplotlib are pre-installed on Colab

# %%
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path.cwd().parent))
from src.config import load_config, ensure_dir, get_device
from src.losses import AsymmetricTemporalPenaltyLoss, WeightedBCELoss
from src.models import build_model
from src.training import train_multi_seed

cfg = load_config()
DATA_ROOT = Path(cfg["paths"]["data_root"])
TENSOR_DIR = DATA_ROOT / cfg["paths"]["tensor_dir"]
PRED_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["predictions_dir"])
RESULTS_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["results_dir"])

print("Device:", get_device())

# %% [markdown]
# ## 1. Load tensors

# %%
def load_split(name: str):
    return {
        f"X_{name}": np.load(TENSOR_DIR / f"X_{name}.npy"),
        f"M_{name}": np.load(TENSOR_DIR / f"M_{name}.npy"),
        f"D_{name}": np.load(TENSOR_DIR / f"D_{name}.npy"),
        f"y_{name}": np.load(TENSOR_DIR / f"y_{name}.npy"),
    }

data = {**load_split("train"), **load_split("val"), **load_split("test")}
F = data["X_train"].shape[-1]
print(f"Feature dim F = {F}")
print(f"Train: {data['X_train'].shape}; positives: {int(data['y_train'][:, 0].sum())}")
print(f"Val:   {data['X_val'].shape}; positives: {int(data['y_val'][:, 0].sum())}")
print(f"Test:  {data['X_test'].shape}; positives: {int(data['y_test'][:, 0].sum())}")

# Compute damped pos_weight for calibration-safe training
from src.losses import compute_pos_weight
pw_method = cfg.get("atp_loss", {}).get("pos_weight_method", "sqrt")
pos_weight = compute_pos_weight(data["y_train"][:, 0], method=pw_method)
train_prev = float(data["y_train"][:, 0].mean())
print(f"pos_weight ({pw_method}): {pos_weight:.2f}")
print(f"train prevalence: {train_prev:.4f}")

# %% [markdown]
# ## 2. XGBoost baseline (single run; no neural seeding)

# %%
X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
y_train_bin = data["y_train"][:, 0].astype(int)
y_test_bin = data["y_test"][:, 0].astype(int)

xgb_clf = xgb.XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    scale_pos_weight=pos_weight,  # same damped weight as neural baselines
    n_jobs=-1, random_state=cfg["split"]["random_seed"],
    eval_metric="aucpr", tree_method="hist",
)
xgb_clf.fit(X_train_flat, y_train_bin)
y_pred_xgb = xgb_clf.predict_proba(X_test_flat)[:, 1]
np.save(PRED_DIR / "xgboost_mean_test_probs.npy", y_pred_xgb)

xgb_metrics = {
    "model": "xgboost",
    "n_seeds": 1,
    "test_pr_auc_mean": float(average_precision_score(y_test_bin, y_pred_xgb)),
    "test_roc_auc_mean": float(roc_auc_score(y_test_bin, y_pred_xgb)),
    "test_brier_mean": float(brier_score_loss(y_test_bin, y_pred_xgb)),
}
print(json.dumps(xgb_metrics, indent=2))
with open(RESULTS_DIR / "xgboost_results.json", "w") as f:
    json.dump(xgb_metrics, f, indent=2)

# %% [markdown]
# ## 3. Neural baselines — multi-seed training
#
# Each model uses ATP loss with the train-set inverse-prevalence weight,
# the same hidden size, batch size, learning rate, dropout, and patience.

# %%
def atp_factory(_seed: int):
    return AsymmetricTemporalPenaltyLoss(
        pos_weight=pos_weight,
        focal_gamma=cfg["atp_loss"]["focal_gamma"],
        temporal_decay_alpha=cfg["atp_loss"]["temporal_decay_alpha"],
        horizon_hours=cfg["atp_loss"]["horizon_hours"],
    )


def model_factory(name: str):
    return lambda _seed: build_model(name, input_dim=F, cfg=cfg, prevalence=train_prev)


# Standard GRU (no Δt anywhere)
train_multi_seed(
    "gru", model_factory("gru"), atp_factory,
    data, cfg, out_dir=PRED_DIR,
)
# Standard GRU + Δt as plain feature (the honest ablation control)
train_multi_seed(
    "gru_delta_features", model_factory("gru_delta_features"), atp_factory,
    data, cfg, out_dir=PRED_DIR,
)
# 1D-CNN
train_multi_seed(
    "cnn1d", model_factory("cnn1d"), atp_factory,
    data, cfg, out_dir=PRED_DIR,
)
# Faithful GRU-D
train_multi_seed(
    "grud", model_factory("grud"), atp_factory,
    data, cfg, out_dir=PRED_DIR,
)

# Move per-model results.json to the results dir
for name in ("gru", "gru_delta_features", "cnn1d", "grud"):
    src = PRED_DIR / f"{name}_results.json"
    if src.exists():
        src.replace(RESULTS_DIR / f"{name}_results.json")

print("\nBaselines complete. Predictions in:", PRED_DIR)
print("Results JSON in:", RESULTS_DIR)
