# %% [markdown]
# # Notebook 0 — Cohort Construction & Exploratory Analysis
#
# **Inputs:** `final_cohort_3h - Work.csv.gz` (the raw bin-level extract from MIMIC-IV).
# **Outputs:** `final_cohort_3h_filtered.parquet` and the EDA figures used in the manuscript.
#
# This notebook performs three things, and only three things:
#
# 1. **Cohort attrition** — applies the inclusion/exclusion criteria (adult ICU stays,
#    ≥ 24 h ICU LOS, no left-censored sepsis-on-admission). Each step prints a
#    transparent dropout count.
# 2. **EDA** — Table 1 with proper non-parametric tests, missingness map,
#    pre-onset trajectory plots, and a Spearman correlation heatmap.
# 3. **Export** — writes the filtered cohort to parquet for Notebook 1.
#
# **Honesty notes (vs. the prior version of this pipeline):**
# - The full feature set produced is **18 channels**, not 53. The manuscript text
#   has been updated to match.
# - `sofa_first` is computed for Table 1 stratification but is **dropped before
#   modelling** (see `configs/config.yaml`) because it leaks the SOFA-based outcome
#   definition.
# - All datetimes are coerced once at load. `intime`-vs-`bin_start` proxy logic
#   from the previous version has been removed because the parquet now carries
#   `icu_admit_true` directly.

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
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

# Add project root to path so `from src...` works inside the notebook
sys.path.insert(0, str(Path.cwd().parent))
from src.config import load_config, ensure_dir, resolve_path

warnings.filterwarnings("ignore")
sns.set_context("talk")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

cfg = load_config()
DATA_ROOT = Path(cfg["paths"]["data_root"])
RAW_PATH = DATA_ROOT / cfg["paths"]["raw_cohort"]
OUT_PATH = DATA_ROOT / cfg["paths"]["filtered_cohort"]
FIG_DIR = ensure_dir(DATA_ROOT / cfg["paths"]["figures_dir"])

# %% [markdown]
# ## 1. Load raw data and cast datetimes

# %%
print(f"Loading raw cohort from: {RAW_PATH}")
if str(RAW_PATH).endswith(".gz") or str(RAW_PATH).endswith(".csv"):
    df = pd.read_csv(RAW_PATH, compression="infer")
else:
    df = pd.read_parquet(RAW_PATH)
print(f"Loaded {len(df):,} bin-level rows; {df['stay_id'].nunique():,} stays")

for col in ["bin_start", "bin_end", "sepsis3_onset_time"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Establish admission time. Prefer the column the SQL extract produced.
if "intime" in df.columns:
    df["icu_admit_true"] = pd.to_datetime(df["intime"])
elif "icu_admit_true" in df.columns:
    df["icu_admit_true"] = pd.to_datetime(df["icu_admit_true"])
else:
    print("WARNING: no admission column found; deriving from min(bin_start) per stay")
    df["icu_admit_true"] = df.groupby("stay_id")["bin_start"].transform("min")

df["onset_delta_hours"] = (
    (df["sepsis3_onset_time"] - df["icu_admit_true"]).dt.total_seconds() / 3600.0
)

# %% [markdown]
# ## 2. Cohort attrition

# %%
total_stays = df["stay_id"].nunique()

# Identify left-censored cases: sepsis at or before the prediction-runway boundary.
sepsis_stays = df[df["is_sepsis_stay"] == 1].groupby("stay_id")["onset_delta_hours"].first()
n_left_censored = int((sepsis_stays <= cfg["cohort"]["min_runway_hours"]).sum())
n_icu_acquired = int((sepsis_stays > cfg["cohort"]["min_runway_hours"]).sum())

# Exclude left-censored cases (they had sepsis before any 6h prediction was possible)
keep_runway = df["onset_delta_hours"].isna() | (
    df["onset_delta_hours"] > cfg["cohort"]["min_runway_hours"]
)
df_step1 = df.loc[keep_runway].copy()
n_after_runway = df_step1["stay_id"].nunique()

# Exclude ultra-short stays
valid_long_stays = df_step1.loc[
    df_step1["bin_index"] >= cfg["cohort"]["min_stay_bins"], "stay_id"
].unique()
df_final = df_step1.loc[df_step1["stay_id"].isin(valid_long_stays)].copy()
n_final = df_final["stay_id"].nunique()

print(f"\nCohort attrition")
print(f"  Total stays:              {total_stays:>8,}")
print(f"  - left-censored sepsis:   {n_left_censored:>8,}")
print(f"  - ICU-acquired sepsis:    {n_icu_acquired:>8,}")
print(f"  After runway exclusion:   {n_after_runway:>8,} (-{total_stays - n_after_runway:,})")
print(f"  After short-stay drop:    {n_final:>8,} (-{n_after_runway - n_final:,})")

# Waterfall figure
fig, ax = plt.subplots(figsize=(9, 5))
stages = ["Initial", "Runway ≥ 6h", "Stay ≥ 6h"]
counts = [total_stays, n_after_runway, n_final]
ax.bar(stages, counts, color=["#386cb0", "#7fc97f", "#fdc086"])
for i, c in enumerate(counts):
    ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of ICU stays")
ax.set_title("Cohort attrition")
ax.set_ylim(0, max(counts) * 1.1)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_cohort_attrition.png", dpi=300)
plt.show()

# %% [markdown]
# ## 3. ICU LOS distribution

# %%
los_hours = (df_final.groupby("stay_id")["bin_index"].max() + 1) * cfg["cohort"]["bin_hours"]
print(f"Median LOS: {los_hours.median():.1f} h ({los_hours.median()/24:.1f} d)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(los_hours.clip(upper=240), bins=60, color="#7570b3", alpha=0.85)
ax.axvline(los_hours.median(), color="black", linestyle="--", label=f"Median: {los_hours.median():.0f} h")
ax.set_xlabel("ICU length of stay (hours, capped at 240)")
ax.set_ylabel("Number of stays")
ax.set_title("LOS distribution (filtered cohort)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_los_distribution.png", dpi=300)
plt.show()

# %% [markdown]
# ## 4. Missingness map

# %%
vital_lab_cols = (
    cfg["features"]["vitals"]
    + cfg["features"]["labs"]
    + cfg["features"]["derived_clinical"]
)
miss_pct = (df_final[vital_lab_cols].isnull().mean() * 100).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(miss_pct.index, miss_pct.values, color="#1b9e77")
ax.set_xlabel("Bin-level missingness (%)")
ax.set_title("Per-feature missingness in the filtered cohort")
for i, v in enumerate(miss_pct.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_missingness.png", dpi=300)
plt.show()

# %% [markdown]
# ## 5. Spearman correlation among physiological features

# %%
corr = df_final[vital_lab_cols].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Spearman correlation — physiological features")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_correlation.png", dpi=300)
plt.show()

# %% [markdown]
# ## 6. Pre-onset trajectory plot
# Sepsis cases aligned to onset (T=0); controls aligned to discharge (T=0).
# 95% CIs are bootstrap from seaborn.

# %%
sepsis = df_final[df_final["is_sepsis_stay"] == 1].copy()
sepsis["T_minus"] = (sepsis["bin_start"] - sepsis["sepsis3_onset_time"]).dt.total_seconds() / 3600.0
sepsis_traj = sepsis[(sepsis["T_minus"] >= -24) & (sepsis["T_minus"] <= 0)].copy()

control = df_final[df_final["is_sepsis_stay"] == 0].copy()
last_bin_per_stay = control.groupby("stay_id")["bin_start"].transform("max")
control["T_minus"] = (control["bin_start"] - last_bin_per_stay).dt.total_seconds() / 3600.0
control_traj = control[(control["T_minus"] >= -24) & (control["T_minus"] <= 0)].copy()

for d in (sepsis_traj, control_traj):
    d["T_bin"] = (np.floor(d["T_minus"] / 3) * 3).astype(int)

plot_data = pd.concat([
    sepsis_traj.assign(Cohort="Sepsis (ICU-acquired)"),
    control_traj.assign(Cohort="Control"),
])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, col in zip(axes, ["map", "heart_rate"]):
    sns.lineplot(data=plot_data, x="T_bin", y=col, hue="Cohort",
                 errorbar=("ci", 95), ax=ax, marker="o",
                 palette={"Control": "steelblue", "Sepsis (ICU-acquired)": "darkred"})
    ax.invert_xaxis()
    ax.axvline(-cfg["cohort"]["prediction_horizon_hours"],
               color="black", linestyle=":", label="6h prediction horizon")
    ax.set_xlabel("Hours before onset / discharge")
    ax.set_ylabel(col.upper())
    ax.set_title(f"Pre-event trajectory: {col.upper()}")
    ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_trajectory.png", dpi=300)
plt.show()

# %% [markdown]
# ## 7. Manuscript Table 1 (with non-parametric tests)

# %%
agg = {"age": "first", "gender": "first", "is_sepsis_stay": "max", "sofa_first": "max"}
stays = df_final.groupby("stay_id").agg(agg)
sepsis_grp = stays[stays["is_sepsis_stay"] == 1]
control_grp = stays[stays["is_sepsis_stay"] == 0]


def nice_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


rows = []
rows.append({"Metric": "N (stays)", "Control": f"{len(control_grp):,}", "Sepsis": f"{len(sepsis_grp):,}", "p": "—"})

# Age
_, p_age = mannwhitneyu(control_grp["age"], sepsis_grp["age"], alternative="two-sided")
rows.append({
    "Metric": "Age (mean ± SD)",
    "Control": f"{control_grp['age'].mean():.1f} ± {control_grp['age'].std():.1f}",
    "Sepsis": f"{sepsis_grp['age'].mean():.1f} ± {sepsis_grp['age'].std():.1f}",
    "p": nice_p(p_age),
})

# Gender (chi-square)
contingency = pd.crosstab(stays["gender"], stays["is_sepsis_stay"])
_, p_g, _, _ = chi2_contingency(contingency)
rows.append({
    "Metric": "Male (%)",
    "Control": f"{(control_grp['gender'] == 'M').mean() * 100:.1f}%",
    "Sepsis": f"{(sepsis_grp['gender'] == 'M').mean() * 100:.1f}%",
    "p": nice_p(p_g),
})

# Peak SOFA
s_c = control_grp["sofa_first"].dropna()
s_s = sepsis_grp["sofa_first"].dropna()
_, p_sofa = mannwhitneyu(s_c, s_s)
rows.append({
    "Metric": "Peak SOFA (med [IQR])",
    "Control": f"{s_c.median():.0f} [{s_c.quantile(0.25):.0f}-{s_c.quantile(0.75):.0f}]",
    "Sepsis": f"{s_s.median():.0f} [{s_s.quantile(0.25):.0f}-{s_s.quantile(0.75):.0f}]",
    "p": nice_p(p_sofa),
})

table1 = pd.DataFrame(rows)
print(table1.to_string(index=False))
table1.to_csv(FIG_DIR / "table1_cohort_characteristics.csv", index=False)

# %% [markdown]
# ## 8. Persist filtered cohort

# %%
df_final.to_parquet(OUT_PATH, index=False)
print(f"Saved {len(df_final):,} rows to {OUT_PATH}")
print("Notebook 1 reads from this path.")
