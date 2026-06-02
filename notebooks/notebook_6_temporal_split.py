# %% [markdown]
# # Temporal Split Validation
#
# This notebook implements a **within-MIMIC-IV temporal split** as a proxy
# for external validation. Instead of random patient-level splitting, patients
# are assigned to train/val/test based on **ICU admission year**:
#
# - **Train:** admissions before the cutoff year (e.g., 2008–2016)
# - **Val:** admissions in the cutoff year (e.g., 2017)
# - **Test:** admissions after the cutoff year (e.g., 2018–2022)
#

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
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))
from src.config import load_config, ensure_dir, set_global_seed
from src.data import (
    encode_gender, censor_post_onset, add_time_to_onset,
    get_feature_columns, get_binary_columns,
    fit_robust_scaler, apply_robust_scaler,
    build_and_save_windows,
)

cfg = load_config()
DATA_ROOT = Path(cfg["paths"]["data_root"])
PARQUET = DATA_ROOT / cfg["paths"]["filtered_cohort"]
TEMPORAL_DIR = ensure_dir(DATA_ROOT / "temporal_split")
TENSOR_DIR = ensure_dir(TEMPORAL_DIR / "tensors")
PRED_DIR = ensure_dir(TEMPORAL_DIR / "predictions")
RESULTS_DIR = ensure_dir(TEMPORAL_DIR / "results")

# %% [markdown]
# ## 1. Load cohort and extract admission year

# %%
df = pd.read_parquet(PARQUET)
df["bin_start"] = pd.to_datetime(df["bin_start"])
df["bin_end"] = pd.to_datetime(df["bin_end"])
df["sepsis3_onset_time"] = pd.to_datetime(df["sepsis3_onset_time"])
df["icu_admit_true"] = pd.to_datetime(df["icu_admit_true"])

# Extract the admission year per stay
admit_year = df.groupby("stay_id")["icu_admit_true"].first().dt.year
df["admit_year"] = df["stay_id"].map(admit_year)

print("Admission year distribution:")
print(admit_year.value_counts().sort_index())
print(f"\nTotal stays: {df['stay_id'].nunique():,}")

# %% [markdown]
# ## 2. Assign temporal split
#
# MIMIC-IV uses **de-identified date-shifted years** (typically 2110–2214,
# NOT 2008–2022). We cannot hardcode calendar years. Instead, we compute
# quantile-based year boundaries from the data to achieve an approximate
# 70% train / 10% val / 20% test split by admission year.
#
# This simulates the realistic deployment scenario: train on the earliest
# 70% of admissions, validate on the next 10%, and test on the most recent 20%.

# %%
# Auto-detect year boundaries from the actual (shifted) admission dates
stay_years = df.groupby("stay_id")["admit_year"].first()
TRAIN_END_YEAR = int(stay_years.quantile(0.70))   # train: years <= this
VAL_END_YEAR   = int(stay_years.quantile(0.80))   # val: years in (TRAIN_END, VAL_END]
                                                    # test: years > VAL_END

def assign_temporal_split(row_year: int) -> str:
    if row_year <= TRAIN_END_YEAR:
        return "train"
    elif row_year <= VAL_END_YEAR:
        return "val"
    else:
        return "test"

df["split"] = df["admit_year"].apply(assign_temporal_split)

split_stats = df.groupby("split")["stay_id"].nunique()
print(f"Detected year range: {int(stay_years.min())}–{int(stay_years.max())} (de-identified)")
print(f"Boundaries: train ≤ {TRAIN_END_YEAR}, val = {TRAIN_END_YEAR+1}–{VAL_END_YEAR}, test ≥ {VAL_END_YEAR+1}")
print("\nTemporal split (stays):")
print(split_stats)

# Check prevalence per split
for s in ["train", "val", "test"]:
    sub = df[df["split"] == s]
    if len(sub) == 0:
        print(f"  {s}: EMPTY — check year boundaries!")
        continue
    prev = sub["sepsis_next_6h"].mean()
    print(f"  {s}: {sub['stay_id'].nunique():,} stays, prevalence = {prev:.4f}")

# %% [markdown]
# ## 3. Tensorize with the temporal split

# %%
TENSOR_DTYPE = np.float32

df = encode_gender(df)
df = censor_post_onset(df)
df = add_time_to_onset(df)

feature_cols = get_feature_columns(cfg)
binary_cols = [c for c in get_binary_columns(cfg) if c in feature_cols]
scale_cols = [c for c in feature_cols if c not in binary_cols]

# CRITICAL: scaling is fitted on TRAIN split only (temporal train)
train_mask = df["split"] == "train"
scaler = fit_robust_scaler(df, train_mask, scale_cols)
df = apply_robust_scaler(df, scale_cols, scaler)
df[scale_cols] = df[scale_cols].fillna(0.0).astype(TENSOR_DTYPE)
df[binary_cols] = df[binary_cols].fillna(0.0).astype(TENSOR_DTYPE)

counts = build_and_save_windows(
    df,
    feature_cols=feature_cols,
    lookback=cfg["cohort"]["lookback_bins"],
    bin_hours=cfg["cohort"]["bin_hours"],
    out_dir=TENSOR_DIR,
)
print("\nTemporal split tensor counts:")
for s, n in counts.items():
    y = np.load(TENSOR_DIR / f"y_{s}.npy", mmap_mode="r")
    print(f"  {s}: {n:,} windows, prevalence = {y[:, 0].mean():.4f}")

# Save feature names and scaling params
np.savez(
    TENSOR_DIR / "scaling_params.npz",
    median=scaler["median"], iqr=scaler["iqr"], columns=scaler["columns"],
)
(TENSOR_DIR / "feature_names.txt").write_text("\n".join(feature_cols))

# %% [markdown]
# ## 4. Train key models on the temporal split
#
# We train the TD-GRU, GRU-D, Standard GRU, and XGBoost only (the four
# most important for the paper's narrative). Multi-seed protocol is the same.

# %%
from src.losses import AsymmetricTemporalPenaltyLoss, compute_pos_weight
from src.models import build_model
from src.training import train_multi_seed
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

def load_split(name: str):
    return {
        f"X_{name}": np.load(TENSOR_DIR / f"X_{name}.npy"),
        f"M_{name}": np.load(TENSOR_DIR / f"M_{name}.npy"),
        f"D_{name}": np.load(TENSOR_DIR / f"D_{name}.npy"),
        f"y_{name}": np.load(TENSOR_DIR / f"y_{name}.npy"),
    }

data = {**load_split("train"), **load_split("val"), **load_split("test")}
F = data["X_train"].shape[-1]
pw_method = cfg.get("atp_loss", {}).get("pos_weight_method", "sqrt")
pos_weight = compute_pos_weight(data["y_train"][:, 0], method=pw_method)
train_prev = float(data["y_train"][:, 0].mean())
print(f"F={F}, pos_weight ({pw_method})={pos_weight:.2f}, train_prev={train_prev:.4f}")


def atp_factory(_seed):
    return AsymmetricTemporalPenaltyLoss(
        pos_weight=pos_weight,
        focal_gamma=cfg["atp_loss"]["focal_gamma"],
        temporal_decay_alpha=cfg["atp_loss"]["temporal_decay_alpha"],
        horizon_hours=cfg["atp_loss"]["horizon_hours"],
    )


# --- TD-GRU ---
train_multi_seed(
    "td_gru_temporal",
    lambda _s: build_model("td_gru", F, cfg, prevalence=train_prev),
    atp_factory, data, cfg, out_dir=PRED_DIR,
)

# --- GRU-D ---
train_multi_seed(
    "grud_temporal",
    lambda _s: build_model("grud", F, cfg, prevalence=train_prev),
    atp_factory, data, cfg, out_dir=PRED_DIR,
)

# --- Standard GRU ---
train_multi_seed(
    "gru_temporal",
    lambda _s: build_model("gru", F, cfg, prevalence=train_prev),
    atp_factory, data, cfg, out_dir=PRED_DIR,
)

# --- XGBoost ---
set_global_seed(42)
X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
y_train_bin = data["y_train"][:, 0].astype(int)
y_test_bin = data["y_test"][:, 0].astype(int)

xgb_clf = xgb.XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    scale_pos_weight=pos_weight, n_jobs=-1, random_state=42,
    eval_metric="aucpr", tree_method="hist",
)
xgb_clf.fit(X_train_flat, y_train_bin)
y_pred_xgb = xgb_clf.predict_proba(X_test_flat)[:, 1]
np.save(PRED_DIR / "xgboost_temporal_mean_test_probs.npy", y_pred_xgb)

print("\nXGBoost (temporal):")
print(f"  PR-AUC: {average_precision_score(y_test_bin, y_pred_xgb):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test_bin, y_pred_xgb):.4f}")
print(f"  Brier: {brier_score_loss(y_test_bin, y_pred_xgb):.4f}")

# %% [markdown]
# ## 5. Comparative table: Random Split vs Temporal Split

# %%
from src.evaluation import bootstrap_ci, paired_bootstrap_pr_auc_test

y_test = data["y_test"][:, 0].astype(int)
tto_test = data["y_test"][:, 1]

MODELS_TEMPORAL = {
    "TD-GRU": "td_gru_temporal",
    "GRU-D": "grud_temporal",
    "Standard GRU": "gru_temporal",
    "XGBoost": "xgboost_temporal",
}

print("\n" + "=" * 80)
print("TEMPORAL SPLIT RESULTS (Supplementary Table)")
print("=" * 80)
print(f"{'Model':<20} {'PR-AUC [95% CI]':<30} {'ROC-AUC [95% CI]':<30} {'Brier':<12}")
print("-" * 80)

temporal_results = {}
for label, stem in MODELS_TEMPORAL.items():
    p = PRED_DIR / f"{stem}_mean_test_probs.npy"
    if not p.exists():
        print(f"  {label}: MISSING")
        continue
    probs = np.load(p)
    pr = bootstrap_ci(y_test, probs, "pr_auc", n_resamples=2000, random_seed=12345)
    roc = bootstrap_ci(y_test, probs, "roc_auc", n_resamples=2000, random_seed=12345)
    bri = brier_score_loss(y_test, probs)
    print(f"{label:<20} {pr['point']:.4f} [{pr['ci_low']:.4f}, {pr['ci_high']:.4f}]"
          f"   {roc['point']:.4f} [{roc['ci_low']:.4f}, {roc['ci_high']:.4f}]"
          f"   {bri:.4f}")
    temporal_results[label] = probs

# Paired bootstrap: TD-GRU vs others
if "TD-GRU" in temporal_results:
    lead = temporal_results["TD-GRU"]
    print(f"\nPaired bootstrap (temporal split, n=2000):")
    for label, probs in temporal_results.items():
        if label == "TD-GRU":
            continue
        res = paired_bootstrap_pr_auc_test(y_test, lead, probs, n_resamples=2000, random_seed=12345)
        sig = "✓" if res["p_value_two_sided"] < 0.05 else "—"
        print(f"  vs {label:<15} Δ={res['obs_diff']:+.4f}  p={res['p_value_two_sided']:.4f}  {sig}")

# PhysioNet utility on temporal split
from src.evaluation import find_best_utility_threshold
print(f"\nPhysioNet utility (temporal split):")
for label, probs in temporal_results.items():
    t, u = find_best_utility_threshold(y_test, probs, tto_test)
    print(f"  {label:<20} threshold={t:.4f}  utility={u:+.4f}")

# Save results
results_summary = {}
for label, probs in temporal_results.items():
    results_summary[label] = {
        "pr_auc": float(average_precision_score(y_test, probs)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
    }
with open(RESULTS_DIR / "temporal_split_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to: {RESULTS_DIR}")

# %%
# === Temporal-split sample sizes  =============================
# Reports the number of windows, unique patients, and sepsis-positive patients
# in each temporal split, addressing the reviewer's request for explicit per-
# split positive counts and prevalence.
# ===========================================================================

for split in ["train", "val", "test"]:
    y = np.load(TENSOR_DIR / f"y_{split}.npy")
    ids = np.load(TENSOR_DIR / f"{split}_row_ids.npy", allow_pickle=True)

    y_bin = y[:, 0].astype(int)

    subjects = np.array([int(str(r).split("_")[0]) for r in ids])

    n_windows = len(y_bin)
    n_pos_windows = int(y_bin.sum())
    n_unique_patients = int(np.unique(subjects).size)
    n_pos_patients = int(np.unique(subjects[y_bin == 1]).size)
    prevalence = n_pos_windows / n_windows

    print(
        f"Temporal {split}: {n_windows:,} windows | "
        f"{n_unique_patients:,} unique patients | "
        f"{n_pos_windows:,} sepsis-positive windows | "
        f"{n_pos_patients:,} sepsis-positive patients | "
        f"window-level prevalence {prevalence:.4%}"
    )

# %%
# === Patient-level cluster bootstrap on temporal-split test set ==============
# Computes 95% CIs for PR-AUC and ROC-AUC using patient-level cluster bootstrap
# on the temporal test set.
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Use notebook-defined temporal tensor and prediction directories
TEMPORAL_TENSOR_DIR = TENSOR_DIR
TEMPORAL_PRED_DIR = PRED_DIR

y_test_temp = np.load(TEMPORAL_TENSOR_DIR / "y_test.npy")
y_temp = y_test_temp[:, 0].astype(int)
tto_temp = y_test_temp[:, 1].astype(float)

ids_temp = np.load(TEMPORAL_TENSOR_DIR / "test_row_ids.npy", allow_pickle=True)
subj_temp = np.array([int(str(r).split("_")[0]) for r in ids_temp])

unique_p_temp = np.unique(subj_temp)
n_p_temp = len(unique_p_temp)

by_p = {int(p): [] for p in unique_p_temp.tolist()}
for w_idx, p in enumerate(subj_temp):
    by_p[int(p)].append(w_idx)

windows_per_p_temp = [
    np.asarray(by_p[int(p)], dtype=np.int64)
    for p in unique_p_temp.tolist()
]

rng_temp = np.random.default_rng(12345)

boot_idx_temp = []
for _ in range(2000):
    sampled_patient_positions = rng_temp.integers(0, n_p_temp, size=n_p_temp)
    sampled_window_indices = np.concatenate(
        [windows_per_p_temp[i] for i in sampled_patient_positions]
    )
    boot_idx_temp.append(sampled_window_indices)

# Load each model's temporal-split predictions
TEMPORAL_MODELS = {
    "TD-GRU":       "td_gru_temporal",
    "GRU-D":        "grud_temporal",
    "Standard GRU": "gru_temporal",
    "XGBoost":      "xgboost_temporal",
}

rows = []

for label, stem in TEMPORAL_MODELS.items():
    p_path = TEMPORAL_PRED_DIR / f"{stem}_mean_test_probs.npy"

    if not p_path.exists():
        print(f"  {label}: missing temporal preds at {p_path}")
        continue

    p = np.load(p_path)

    if len(p) != len(y_temp):
        print(
            f"  {label}: prediction length mismatch. "
            f"Expected {len(y_temp)}, got {len(p)}"
        )
        continue

    pr_s, roc_s = [], []

    for idx in boot_idx_temp:
        y_b = y_temp[idx]
        p_b = p[idx]

        # Need both classes for ROC-AUC, and at least one positive for PR-AUC
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue

        pr_s.append(float(average_precision_score(y_b, p_b)))
        roc_s.append(float(roc_auc_score(y_b, p_b)))

    rows.append({
        "Model": label,
        "PR-AUC": (
            f"{average_precision_score(y_temp, p):.4f} "
            f"[{np.percentile(pr_s, 2.5):.4f}, {np.percentile(pr_s, 97.5):.4f}]"
        ),
        "ROC-AUC": (
            f"{roc_auc_score(y_temp, p):.4f} "
            f"[{np.percentile(roc_s, 2.5):.4f}, {np.percentile(roc_s, 97.5):.4f}]"
        ),
        "Valid bootstrap samples": len(pr_s),
    })

temp_ci = pd.DataFrame(rows)

print("\nTemporal-split discrimination with patient-level cluster CIs:")
print(temp_ci.to_string(index=False))

temp_ci.to_csv(RESULTS_DIR / "temporal_split_patientCI.csv", index=False)
print(f"\nSaved to: {RESULTS_DIR / 'temporal_split_patientCI.csv'}")
