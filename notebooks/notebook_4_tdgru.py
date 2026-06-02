# %% [markdown]
# # Notebook 4 — TD-GRU (Lead Model), Ablations, and Hyperparameter Sweep
#
# Three things happen here:
#
# 1. **Lead model** — TD-GRU + ATP loss, multi-seed.
# 2. **Ablations** — three controlled variants:
#    - Architecture ablation: Standard GRU + Δt-as-feature (already trained in Notebook 2;
#      re-loaded here for the table)
#    - Loss ablation: TD-GRU + Weighted BCE (no focal, no temporal)
#    - Joint ablation: Standard GRU + ATP (re-loaded from Notebook 2)
# 3. **ATP hyperparameter sensitivity sweep** — small grid over (γ, α) reported on
#    a single seed for the table; the final lead-model values come from the
#    sensitivity-best configuration on the validation set.
#
# All ablations use identical training budgets to the lead model.

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
import threading, time
def _drive_keepalive():
    while True:
        try:
            os.listdir(PROJECT_ROOT)  # touch Drive every 5 min
        except: pass
        time.sleep(300)
threading.Thread(target=_drive_keepalive, daemon=True).start()

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))
from src.config import load_config, ensure_dir, get_device, set_global_seed
from src.losses import AsymmetricTemporalPenaltyLoss, WeightedBCELoss
from src.models import build_model
from src.training import build_loader, train_one_run, train_multi_seed
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

cfg = load_config()
DATA_ROOT = Path(cfg["paths"]["data_root"])
TENSOR_DIR = DATA_ROOT / cfg["paths"]["tensor_dir"]
PRED_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["predictions_dir"])
RESULTS_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["results_dir"])
device = get_device()
print("Device:", device)

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
print(f"F={F}, pos_weight ({pw_method})={pos_weight:.2f}, prevalence={train_prev:.4f}")

# %% [markdown]
# ## 1. Lead model: TD-GRU + ATP loss (multi-seed)

# %%
def atp_factory(_seed: int):
    return AsymmetricTemporalPenaltyLoss(
        pos_weight=pos_weight,
        focal_gamma=cfg["atp_loss"]["focal_gamma"],
        temporal_decay_alpha=cfg["atp_loss"]["temporal_decay_alpha"],
        horizon_hours=cfg["atp_loss"]["horizon_hours"],
    )

def td_gru_factory(_seed: int):
    return build_model("td_gru", input_dim=F, cfg=cfg, prevalence=train_prev)

train_multi_seed(
    "td_gru", td_gru_factory, atp_factory,
    data, cfg, out_dir=PRED_DIR,
)
src = PRED_DIR / "td_gru_results.json"
if src.exists():
    src.replace(RESULTS_DIR / "td_gru_results.json")

# %% [markdown]
# ## 2. Ablation: TD-GRU + Weighted BCE (loss-only ablation)

# %%
def bce_factory(_seed: int):
    return WeightedBCELoss(pos_weight=pos_weight)

train_multi_seed(
    "td_gru_bce", td_gru_factory, bce_factory,
    data, cfg, out_dir=PRED_DIR,
)
src = PRED_DIR / "td_gru_bce_results.json"
if src.exists():
    src.replace(RESULTS_DIR / "td_gru_bce_results.json")

# %% [markdown]
# ## 3. Architecture ablations
# These were already trained in Notebook 2:
# - `gru` (Standard GRU + ATP, Δt unseen)
# - `gru_delta_features` (Standard GRU + ATP, Δt as plain feature)
#
# We just load their results.json for the consolidated ablation table below.

# %%
def load_results(name: str):
    p = RESULTS_DIR / f"{name}_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


variants = {
    "TD-GRU + ATP (full)":             "td_gru",
    "TD-GRU + Weighted BCE (no ATP)":  "td_gru_bce",
    "Standard GRU + ATP (no decay)":   "gru",
    "Standard GRU + Δt-feat + ATP":    "gru_delta_features",
}

ab_rows = []
for label, name in variants.items():
    r = load_results(name)
    if r is None:
        ab_rows.append({"Variant": label, "PR-AUC": "—", "ROC-AUC": "—", "Brier": "—"})
        continue
    ab_rows.append({
        "Variant": label,
        "PR-AUC":  f"{r['test_pr_auc_mean']:.4f} ± {r.get('test_pr_auc_std', 0):.4f}",
        "ROC-AUC": f"{r['test_roc_auc_mean']:.4f} ± {r.get('test_roc_auc_std', 0):.4f}",
        "Brier":   f"{r['test_brier_mean']:.4f} ± {r.get('test_brier_std', 0):.4f}",
    })

ab_df = pd.DataFrame(ab_rows)
print(ab_df.to_string(index=False))
ab_df.to_csv(RESULTS_DIR / "ablation_table.csv", index=False)

# %% [markdown]
# ## 4. ATP hyperparameter sensitivity sweep
#
# Single seed, small grid. Reported on **validation** PR-AUC so we never tune on test.
# After the sweep, the best (γ, α) combo can be promoted to the manuscript headline
# (or you can just stick with the defaults if they're within noise of the best).

# %%
GAMMA_GRID = [0.0, 0.5, 1.5, 3.0]
ALPHA_GRID = [0.0, 0.1, 0.5, 1.0]
SEED = cfg["seeds"][0]

# Use a smaller training budget for the sweep so it's tractable.
# This is for *exploration only*; the final lead model uses the chosen config
# with the full multi-seed protocol above.
sweep_cfg = json.loads(json.dumps(cfg))  # deep copy
sweep_cfg["training"]["max_epochs"] = max(8, cfg["training"]["max_epochs"] // 3)
sweep_cfg["training"]["early_stopping_patience"] = 3

sweep_rows = []
for g in GAMMA_GRID:
    for a in ALPHA_GRID:
        set_global_seed(SEED)
        model = build_model("td_gru", input_dim=F, cfg=sweep_cfg, prevalence=train_prev)
        loss = AsymmetricTemporalPenaltyLoss(
            pos_weight=pos_weight, focal_gamma=g, temporal_decay_alpha=a,
            horizon_hours=cfg["atp_loss"]["horizon_hours"],
        )
        train_loader = build_loader(
            data["X_train"], data["M_train"], data["D_train"], data["y_train"],
            batch_size=cfg["training"]["batch_size"], shuffle=True,
        )
        val_loader = build_loader(
            data["X_val"], data["M_val"], data["D_val"], data["y_val"],
            batch_size=cfg["training"]["batch_size"], shuffle=False,
        )
        _, hist = train_one_run(
            model, loss, train_loader, val_loader,
            sweep_cfg, SEED, log_prefix=f"[sweep g={g} a={a}] ",
        )
        sweep_rows.append({"gamma": g, "alpha": a, "best_val_pr_auc": hist["best_val_pr_auc"]})

sweep_df = pd.DataFrame(sweep_rows).sort_values("best_val_pr_auc", ascending=False)
print(sweep_df.to_string(index=False))
sweep_df.to_csv(RESULTS_DIR / "atp_sensitivity.csv", index=False)
print(f"\nBest config: γ={sweep_df.iloc[0]['gamma']}, α={sweep_df.iloc[0]['alpha']}, "
      f"val PR-AUC={sweep_df.iloc[0]['best_val_pr_auc']:.4f}")
