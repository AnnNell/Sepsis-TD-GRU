# %% [markdown]
# # Notebook 3 — Transformer Baseline (Continuous-Time PE)
#
# A multi-head self-attention encoder. Unlike the prior pipeline's vanilla
# Transformer, this version receives **Δt as a learned positional encoding**,
# making it a fair comparator against TD-GRU on irregular sequences.
#
# Architecture: 2 encoder blocks × 4 heads, hidden dim = `cfg.training.hidden_size`,
# `feedforward_dim = 4 * hidden`. Same training budget as Notebook 2.
#
# This is its own notebook because Transformer training is sometimes unstable
# under extreme class imbalance and benefits from inspecting loss curves
# separately. If you want to fold it into Notebook 2, just import `train_multi_seed`
# with `name='transformer'`.

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

sys.path.insert(0, str(Path.cwd().parent))
from src.config import load_config, ensure_dir, get_device
from src.losses import AsymmetricTemporalPenaltyLoss
from src.models import build_model
from src.training import train_multi_seed

cfg = load_config()
DATA_ROOT = Path(cfg["paths"]["data_root"])
TENSOR_DIR = DATA_ROOT / cfg["paths"]["tensor_dir"]
PRED_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["predictions_dir"])
RESULTS_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["results_dir"])

print("Device:", get_device())

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
from src.losses import compute_pos_weight
pw_method = cfg.get("atp_loss", {}).get("pos_weight_method", "sqrt")
pos_weight = compute_pos_weight(data["y_train"][:, 0], method=pw_method)
train_prev = float(data["y_train"][:, 0].mean())


def atp_factory(_seed: int):
    return AsymmetricTemporalPenaltyLoss(
        pos_weight=pos_weight,
        focal_gamma=cfg["atp_loss"]["focal_gamma"],
        temporal_decay_alpha=cfg["atp_loss"]["temporal_decay_alpha"],
        horizon_hours=cfg["atp_loss"]["horizon_hours"],
    )


def model_factory(_seed: int):
    return build_model("transformer", input_dim=F, cfg=cfg, prevalence=train_prev)

# %%
train_multi_seed(
    "transformer", model_factory, atp_factory,
    data, cfg, out_dir=PRED_DIR,
)

src = PRED_DIR / "transformer_results.json"
if src.exists():
    src.replace(RESULTS_DIR / "transformer_results.json")
print("Transformer training complete.")
