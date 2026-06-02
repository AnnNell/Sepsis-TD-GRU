# %% [markdown]
# # Notebook 1 — Sequence Tensorization
#
# Convert the bin-level filtered cohort into model-ready tensors.
#
# **What this notebook produces (per split):**
# - `X_{split}.npy`  shape `(N, 8, 18)` — scaled features, zero-imputed in z-space
# - `M_{split}.npy`  shape `(N, 8, 18)` — observation mask (1 if observed, 0 otherwise)
# - `D_{split}.npy`  shape `(N, 8, 18)` — Δt in hours since each feature was last observed
# - `y_{split}.npy`  shape `(N, 2)`     — `[label, time_to_onset_hours]`
# - `{split}_row_ids.npy`               — string IDs `<subject>_<stay>_<bin>`

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
from src.config import load_config, ensure_dir
from src.data import tensorize

cfg = load_config()
DATA_ROOT = Path(cfg["paths"]["data_root"])
PARQUET = DATA_ROOT / cfg["paths"]["filtered_cohort"]
TENSOR_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["tensor_dir"])

print(f"Reading cohort:  {PARQUET}")
print(f"Writing tensors: {TENSOR_DIR}")

# %% [markdown]
# ## 1. Run the end-to-end tensorizer

# %%
summary = tensorize(cfg, PARQUET, TENSOR_DIR)
print(json.dumps({k: v for k, v in summary.items() if k != "feature_cols"}, indent=2))
print("\nFeature order:")
for i, f in enumerate(summary["feature_cols"]):
    print(f"  {i:2d}  {f}")

# %% [markdown]
# ## 2. Sanity checks
#
# These checks must pass for the rest of the pipeline to be trustworthy.

# %%
X_train = np.load(TENSOR_DIR / "X_train.npy", mmap_mode="r")
M_train = np.load(TENSOR_DIR / "M_train.npy", mmap_mode="r")
D_train = np.load(TENSOR_DIR / "D_train.npy", mmap_mode="r")
y_train = np.load(TENSOR_DIR / "y_train.npy", mmap_mode="r")
X_test = np.load(TENSOR_DIR / "X_test.npy", mmap_mode="r")
y_test = np.load(TENSOR_DIR / "y_test.npy", mmap_mode="r")
test_ids = np.load(TENSOR_DIR / "test_row_ids.npy")

# Shape consistency
T = cfg["cohort"]["lookback_bins"]
F = len(summary["feature_cols"])
assert X_train.shape[1:] == (T, F), f"Bad X_train shape: {X_train.shape}"
assert M_train.shape == X_train.shape
assert D_train.shape == X_train.shape
assert y_train.shape[1] == 2
print("Shape checks passed.")

# Δt: must be > 0 everywhere (no zeros at observed bins, the prior bug)
assert (D_train > 0).all(), "Δt has zeros — boundary fix not applied"
print(f"Δt range: [{float(D_train.min()):.2f}, {float(D_train.max()):.2f}] hours")

# Mask values are exactly {0, 1}
unique_mask = np.unique(M_train[:1000])  # subset for memory
assert set(unique_mask.tolist()) <= {0.0, 1.0}, f"Mask not binary: {unique_mask}"
print("Mask is binary.")

# Time-to-onset for negatives is the sentinel value; for positives it is in [-?, +H]
neg_tto = y_train[y_train[:, 0] == 0, 1]
pos_tto = y_train[y_train[:, 0] == 1, 1]
print(f"Negative tto (sentinel): {np.unique(neg_tto)[:3]} ...")
print(f"Positive tto: min={pos_tto.min():.2f} max={pos_tto.max():.2f} median={np.median(pos_tto):.2f}")

# Test row id alignment
assert len(test_ids) == len(X_test) == len(y_test)
print(f"Test row IDs aligned ({len(test_ids):,} samples)")
print(f"Archival hash: {summary['test_row_id_hash']}")
