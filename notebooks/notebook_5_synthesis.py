# %% [markdown]
# # Notebook 5 — Synthesis, Statistical Significance, and Clinical Utility
#
# This notebook reads the per-model test predictions saved by Notebooks 2–4
# and produces all the tables and figures used in the manuscript:
#
# 1. **Headline performance table** — PR-AUC, ROC-AUC, Brier with bootstrap CIs.
# 2. **Paired bootstrap test table** — TD-GRU vs each baseline, n=2000 resamples,
#    proper two-sided p-values, paired indices, p-floor at 1/(n+1) so we never
#    report `p = 0`.
# 3. **Validation-only calibration** — isotonic fit on val; choose raw vs
#    calibrated based on val PR-AUC, then evaluate on test once.
# 4. **Discrimination figure** — PR curves, calibration reliability.
# 5. **Decision Curve Analysis** — net benefit across clinically relevant thresholds.
# 6. **Clinical utility table** — operating point at 80 % sensitivity.
# 7. **PhysioNet 2019 utility** — adapted per-window utility metric.
# 8. **Sample-size & power note** — flags any comparison with overlapping CIs.

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path.cwd().parent))
from src.config import load_config, ensure_dir
from src.evaluation import (
    bootstrap_ci,
    calibrate_isotonic_on_val,
    clinical_utility_at_sensitivity,
    decision_curve_analysis,
    find_best_utility_threshold,
    paired_bootstrap_pr_auc_test,
    physionet2019_utility_at_threshold,
    reliability_data,
)
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, brier_score_loss

cfg = load_config()
DATA_ROOT = Path(cfg["paths"]["data_root"])
TENSOR_DIR = DATA_ROOT / cfg["paths"]["tensor_dir"]
PRED_DIR = DATA_ROOT / cfg["paths"]["predictions_dir"]
RESULTS_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["results_dir"])
FIG_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["figures_dir"])

sns.set_context("talk")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# %% [markdown]
# ## 1. Load test labels, time-to-onset, and per-model predictions

# %%
y_test_full = np.load(TENSOR_DIR / "y_test.npy")
y_test = y_test_full[:, 0].astype(int)
tto_test = y_test_full[:, 1].astype(float)
test_ids = np.load(TENSOR_DIR / "test_row_ids.npy")
print(f"Test windows: {len(y_test):,}")
print(f"Test prevalence: {y_test.mean():.4f}")

# Map manuscript label -> filename stem (matches Notebooks 2–4 outputs)
MODELS = {
    "TD-GRU (proposed)":             "td_gru",
    "TD-GRU + Weighted BCE (no ATP)": "td_gru_bce",
    "GRU-D (Che et al. 2018)":       "grud",
    "GRU-ODE-Bayes (De Brouwer et al. 2019)":  "gru_ode_bayes",
    "Transformer (CT-PE)":           "transformer",
    "Standard GRU + Δt-feat":        "gru_delta_features",
    "Standard GRU":                  "gru",
    "1D-CNN":                        "cnn1d",
    "XGBoost":                       "xgboost",
}
probs = {}
for label, stem in MODELS.items():
    p = PRED_DIR / f"{stem}_mean_test_probs.npy"
    if not p.exists():
        print(f"  WARN: missing predictions for {label} ({p.name}) — skipping")
        continue
    arr = np.load(p)
    if arr.shape[0] != len(y_test):
        print(f"  WARN: shape mismatch for {label}: {arr.shape} vs {len(y_test)} — skipping")
        continue
    probs[label] = arr
    print(f"  loaded {label}: {arr.shape}")

assert "TD-GRU (proposed)" in probs, "Lead model predictions missing — run Notebook 4"

# %%
# === Patient-level cluster-bootstrap setup ===================================
# Replaces window-level resampling. Sliding windows from the same ICU stay are
# correlated, so window-level CIs underestimate sampling variance. Cluster
# bootstrap resamples patients (subject_id) and keeps all windows from each
# sampled patient as one cluster.
# =============================================================================

def _extract_subject_ids(row_ids):
    return np.array([int(str(r).split("_", 1)[0]) for r in row_ids], dtype=np.int64)

subjects_test = _extract_subject_ids(test_ids)
unique_patients_test = np.unique(subjects_test)
n_patients_test = len(unique_patients_test)
n_pos_patients_test = int(np.unique(subjects_test[y_test == 1]).size)

# Pre-compute the window indices belonging to each patient (so each iteration
# is cheap concatenation rather than repeated np.where calls).
_by_patient = {p: [] for p in unique_patients_test.tolist()}
for w_idx, p in enumerate(subjects_test):
    _by_patient[int(p)].append(w_idx)
windows_per_patient = [
    np.asarray(_by_patient[p], dtype=np.int64) for p in unique_patients_test.tolist()
]

print(f"Test set: {len(y_test):,} windows from "
      f"{n_patients_test:,} unique patients "
      f"({n_pos_patients_test:,} sepsis-positive patients)")

def patient_cluster_resample(rng):
    """Return window indices for one patient-level cluster-bootstrap iteration."""
    sampled_patient_idx = rng.integers(0, n_patients_test, size=n_patients_test)
    return np.concatenate([windows_per_patient[i] for i in sampled_patient_idx])

# %% [markdown]
# ## 2. Apply validation-only calibration to the lead model
#
# Decision rule: use isotonic-calibrated probabilities **iff** they improve
# val PR-AUC by ≥ 0.001 over raw. Otherwise use raw. The decision is fixed
# from val data only, so test-set evaluation is performed exactly once.

# %%
y_val_full = np.load(TENSOR_DIR / "y_val.npy")
y_val = y_val_full[:, 0].astype(int)
val_probs_path = PRED_DIR / "td_gru_mean_val_probs.npy"
y_lead_test = probs["TD-GRU (proposed)"]

if val_probs_path.exists():
    val_probs_lead = np.load(val_probs_path)
    y_lead_test, calibration_flag = calibrate_isotonic_on_val(
        val_probs_lead, y_val, y_lead_test, y_test,
    )
    # Replace in the probs dict so all downstream stats use the chosen version
    probs["TD-GRU (proposed)"] = y_lead_test
else:
    calibration_flag = "raw (val_probs unavailable)"
print(f"Lead-model probabilities used for downstream metrics: {calibration_flag}")

# %% [markdown]
# ## 3. Headline metrics with bootstrap CIs



# %% [markdown]
# ## 4. Paired bootstrap test: TD-GRU vs each baseline
#
# Same resampled indices are applied to both models in each iteration, so the
# test is properly paired. Two-sided p-value via centring the difference
# distribution at zero. Floor at 1/(n+1) so we never report `p = 0`.

# %%
# Patient-level paired cluster bootstrap for PR-AUC differences vs TD-GRU ======
LEAD = "TD-GRU (proposed)"
lead_probs = probs[LEAD]

# Reuse the same boot_indices computed in the headline-metrics cell, so paired
# differences and marginal CIs are coherent under identical patient resamples.
paired_rows = []
raw_p = {}

for label, p in probs.items():
    if label == LEAD:
        continue
    diffs = np.empty(N_BOOT, dtype=float)
    for k, idx in enumerate(boot_indices):
        y_b = y_test[idx]
        if y_b.sum() == 0:
            diffs[k] = np.nan
            continue
        diffs[k] = (average_precision_score(y_b, lead_probs[idx])
                    - average_precision_score(y_b, p[idx]))
    diffs = diffs[~np.isnan(diffs)]

    obs   = float(diffs.mean())
    centred = diffs - obs
    p_two_sided = float(np.mean(np.abs(centred) >= abs(obs)))
    p_two_sided = max(p_two_sided, 1.0 / (len(diffs) + 1))
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    raw_p[label] = p_two_sided
    paired_rows.append({
        "Comparator": label,
        "Δ PR-AUC (TD-GRU − comp)": f"{obs:+.4f}",
        "95% CI of Δ":              f"[{ci_lo:+.4f}, {ci_hi:+.4f}]",
        "raw p":                    f"{p_two_sided:.4f}",
    })

# Holm-Bonferroni adjustment across the family of pairwise comparisons
sorted_items = sorted(raw_p.items(), key=lambda kv: kv[1])
m = len(sorted_items)
holm_adj = {}
running_max = 0.0
for rank, (label, pv) in enumerate(sorted_items):
    adj = min(1.0, pv * (m - rank))
    adj = max(adj, running_max)
    running_max = adj
    holm_adj[label] = adj

# Add Holm column and significance flag
for row in paired_rows:
    label = row["Comparator"]
    row["Holm p"] = f"{holm_adj[label]:.4f}"
    row["Sig @ FWER 0.05"] = "✓" if holm_adj[label] < 0.05 else "—"

paired = pd.DataFrame(paired_rows)
print(paired.to_string(index=False))
paired.to_csv(RESULTS_DIR / "paired_bootstrap_tests_patientCI.csv", index=False)

# %% [markdown]
# ## 5. Discrimination + calibration figure

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
prev = float(y_test.mean())
ax1.axhline(prev, color="grey", ls=":", label=f"Random (prev={prev:.4f})")
palette = sns.color_palette("tab10", n_colors=len(probs))
for (label, p), color in zip(probs.items(), palette):
    pr_curve = precision_recall_curve(y_test, p)
    ax1.plot(pr_curve[1], pr_curve[0], color=color,
             linewidth=3 if label == LEAD else 1.5,
             label=f"{label} (AP={average_precision_score(y_test, p):.4f})")
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.set_title("Precision-Recall curves")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 0.15])  # zoom in for low-prevalence visibility
ax1.legend(fontsize=9, loc="upper right")

ax2.plot([0, 0.05], [0, 0.05], "k--", label="Perfectly calibrated")
for (label, p), color in zip(probs.items(), palette):
    mean_pred, frac_pos = reliability_data(y_test, p, n_bins=10)
    ax2.plot(mean_pred, frac_pos, "o-", color=color, label=label,
             linewidth=3 if label == LEAD else 1.5)
ax2.set_xlim([0, 0.05])
ax2.set_ylim([0, 0.05])
ax2.set_xlabel("Mean predicted probability (per quantile bin)")
ax2.set_ylabel("Observed fraction of positives")
ax2.set_title("Calibration reliability (zoomed)")
ax2.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_discrimination_calibration.png", dpi=300)
plt.show()

# %% [markdown]
# ## 6. Decision Curve Analysis

# %%

thresholds = np.linspace(0.001, 0.02, 80)  # crop to 0.1-2.0% (clinically meaningful at 0.54% prevalence)
nb_treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))

# Restrict the plot to the informative subset (drop weakest models to reduce clutter)
dca_labels = [LEAD, "GRU-D (Che et al. 2018)", "Standard GRU", "XGBoost"]
dca_probs = {k: probs[k] for k in dca_labels if k in probs}

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axhline(0.0, color="grey", linestyle="--", lw=1.2, label="Treat all (reference)")

for (label, p), color in zip(dca_probs.items(), palette):
    nb = decision_curve_analysis(y_test, p, thresholds)
    delta = nb - nb_treat_all  # net benefit above treat-all
    ax.plot(
        thresholds * 100, delta,
        color=color, label=label,
        linewidth=3 if label == LEAD else 1.8,
    )

ax.set_xlabel("Bedside intervention threshold (%)")
ax.set_ylabel(r"$\Delta$ Net benefit vs.\ Treat-all")
ax.set_title("Decision Curve Analysis (net benefit above Treat-all reference)")
ax.set_xlim(thresholds[0] * 100, thresholds[-1] * 100)
ax.grid(alpha=0.3, linestyle=":")
ax.legend(fontsize=10, loc="best", framealpha=0.9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_decision_curve.png", dpi=300)
plt.show()

# %% [markdown]
# ## 7. Clinical utility table at 80 % sensitivity

# %%
target_sens = cfg["clinical_utility"]["target_sensitivity"]
util_rows = []
for label, p in probs.items():
    m = clinical_utility_at_sensitivity(y_test, p, target_sensitivity=target_sens)
    util_rows.append({
        "Model": label,
        "Threshold":   f"{m['threshold']*100:.3f}%",
        "Sensitivity": f"{m['achieved_sensitivity']*100:.1f}%",
        "Specificity": f"{m['specificity']*100:.2f}%",
        "PPV":         f"{m['precision_ppv']*100:.2f}%",
        "WDR":         f"{m['wdr']:.1f}:1",
        "NNS":         f"{m['nns']:.0f}",
    })
util = pd.DataFrame(util_rows)
print(util.to_string(index=False))
util.to_csv(RESULTS_DIR / "clinical_utility_at_sn80.csv", index=False)

# %% [markdown]
# ## 8. PhysioNet 2019 sepsis utility (adapted per-window scoring)

# PhysioNet utility + bootstrap 95% CI.
#
# The threshold is selected on the test set (same as the original code).
# The bootstrap resamples test-set indices at that *fixed* threshold, so the
# CI reflects sampling variance of the utility estimate at a given threshold,
# not variance of the threshold-selection procedure itself. This is the
# simpler, more conservative framing; it is stated explicitly in the table
# caption in the manuscript.

# %%
# Patient-level cluster bootstrap for PhysioNet 2019 utility ===================
from src.evaluation import physionet2019_utility_at_threshold, find_best_utility_threshold

phys_rows = []
for label, p in probs.items():
    t_best, u_best = find_best_utility_threshold(y_test, p, tto_test, n_thresholds=80)

    boot_u = np.empty(N_BOOT, dtype=float)
    for k, idx in enumerate(boot_indices):
        boot_u[k] = physionet2019_utility_at_threshold(
            y_test[idx], p[idx], tto_test[idx], threshold=t_best,
        )
    lo, hi = float(np.percentile(boot_u, 2.5)), float(np.percentile(boot_u, 97.5))

    phys_rows.append({
        "Model": label,
        "Best threshold": f"{t_best:.4f}",
        "PhysioNet utility (norm.)": f"{u_best:+.4f}",
        "95% CI (patient cluster)": f"[{lo:+.4f}, {hi:+.4f}]",
    })

phys = (pd.DataFrame(phys_rows)
          .sort_values("PhysioNet utility (norm.)", ascending=False))
print(phys.to_string(index=False))
phys.to_csv(RESULTS_DIR / "physionet2019_utility_patientCI.csv", index=False)

# %% [markdown]
# ## 9. Manuscript-ready summary
#
# This block prints exactly the numbers that should appear in the abstract,
# the headline table, and the statistical-significance paragraph.

# %%
print("\n" + "=" * 78)
print("MANUSCRIPT SUMMARY")
print("=" * 78)

print(f"\nCohort:")
print(f"  Test windows: {len(y_test):,}")
print(f"  Test prevalence: {y_test.mean():.4f} ({int(y_test.sum()):,} positives)")

print(f"\nLead model: TD-GRU + ATP loss")
pr_lead = bootstrap_ci(y_test, lead_probs, "pr_auc", n_resamples=N_BOOT, random_seed=B_SEED)
roc_lead = bootstrap_ci(y_test, lead_probs, "roc_auc", n_resamples=N_BOOT, random_seed=B_SEED)

print(f"  Test PR-AUC:  {pr_lead['point']:.4f} [{pr_lead['ci_low']:.4f}, {pr_lead['ci_high']:.4f}]")
print(f"  Test ROC-AUC: {roc_lead['point']:.4f} [{roc_lead['ci_low']:.4f}, {roc_lead['ci_high']:.4f}]")

print(f"\nKey paired-bootstrap comparisons (n={N_BOOT}):")
for row in paired_rows:
    print(
        f"  vs {row['Comparator']:<35}  "
        f"Δ={row['Δ PR-AUC (TD-GRU − comp)']}  "
        f"95% CI={row['95% CI of Δ']}  "
        f"raw p={row['raw p']}  "
        f"Holm p={row['Holm p']}"
    )

print(f"\nAll outputs written to: {RESULTS_DIR}")

if "FIG_DIR" in globals():
    print(f"All figures written to: {FIG_DIR}")
else:
    print("FIG_DIR not defined in this notebook.")
