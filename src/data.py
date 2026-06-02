"""
Data preprocessing: convert the bin-level cohort parquet into model-ready tensors.

This module fixes the following issues from the prior pipeline:

1. **Honest feature count.** Uses the 18 features actually present in
   the parquet (vitals + labs + interventions + demographics).
   Excludes `sofa_first` to avoid label leakage.

2. **Correct Δt boundary.** Δt at an observed bin is set to `bin_hours`
   (i.e. one bin since "now"), not 0. First-bin Δt is also `bin_hours`.
   This matches Che et al. 2018 (GRU-D) and avoids the pathological
   gamma=1 collapse at observed timesteps.

3. **Train-only scaling.** Robust (median/IQR) scaling fitted on TRAIN
   split only, applied to all splits.

4. **No within-stay leakage.** Strict 8-bin windows, no padding.
   Patient-level split column (`split`) carried through.

5. **Archival hash on test row IDs** retained (kept simple — no theatre).

Tensors produced
----------------
For each split in {train, val, test}:
    X_{split}.npy   shape (N, T, F)   raw scaled features (zero-imputed)
    M_{split}.npy   shape (N, T, F)   1.0 if observed, 0.0 if missing
    D_{split}.npy   shape (N, T, F)   Δt in hours since last obs of that feature
    y_{split}.npy   shape (N, 2)      [label, time_to_onset_hours]
    {split}_row_ids.npy   shape (N,)  string IDs for joining predictions

Also writes:
    feature_names.txt     ordered feature list
    scaling_params.npz    medians, iqrs (train only)
    cohort_summary.json   sizes, prevalences, hash
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TENSOR_DTYPE = np.float32


# -----------------------------------------------------------------------------
# Feature selection
# -----------------------------------------------------------------------------
def get_feature_columns(cfg: Dict) -> List[str]:
    """Return the ordered list of feature column names used by all models.

    Honest reporting: this is exactly 18 features as backed by the parquet.
    Categorical `gender` is encoded as binary (M=0, F=1).
    """
    f = cfg["features"]
    cols: List[str] = []
    cols += list(f["vitals"])
    cols += list(f["labs"])
    cols += list(f["derived_clinical"])
    cols += list(f["binary_interventions"])
    cols += list(f["demographics"])
    cols += list(f["static_severity"])
    drop = set(f.get("drop_for_modeling", []))
    cols = [c for c in cols if c not in drop]
    return cols


def get_binary_columns(cfg: Dict) -> List[str]:
    """Columns that should NOT be standardized (binary or categorical)."""
    return list(cfg["features"]["binary_interventions"]) + ["gender"]


# -----------------------------------------------------------------------------
# Scaling (train-only)
# -----------------------------------------------------------------------------
def fit_robust_scaler(
    df: pd.DataFrame, train_mask: pd.Series, columns: List[str]
) -> Dict[str, np.ndarray]:
    """Fit median and IQR on TRAIN split only.

    NaNs are skipped during fitting so they do not bias the median/IQR.
    """
    train = df.loc[train_mask, columns]
    med = train.median(skipna=True).to_numpy(dtype=TENSOR_DTYPE)
    q1 = train.quantile(0.25).to_numpy(dtype=TENSOR_DTYPE)
    q3 = train.quantile(0.75).to_numpy(dtype=TENSOR_DTYPE)
    iqr = (q3 - q1)
    iqr[iqr == 0.0] = 1.0  # avoid division by zero
    return {"median": med, "iqr": iqr.astype(TENSOR_DTYPE), "columns": np.array(columns)}


def apply_robust_scaler(
    df: pd.DataFrame, columns: List[str], scaler: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Apply (x - median) / IQR. Preserves NaN."""
    df = df.copy()
    df[columns] = (df[columns].to_numpy(dtype=TENSOR_DTYPE) - scaler["median"]) / scaler["iqr"]
    return df


# -----------------------------------------------------------------------------
# Mask construction
# -----------------------------------------------------------------------------
def build_observation_mask(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Return 0/1 mask: 1 if observed in this bin, 0 if missing.

    Uses the pre-computed `<feat>_missing` columns when available (they are
    in the parquet), else falls back to .notna().
    """
    n = len(df)
    F = len(feature_cols)
    mask = np.ones((n, F), dtype=TENSOR_DTYPE)
    for j, col in enumerate(feature_cols):
        miss_col = f"{col}_missing"
        if miss_col in df.columns:
            mask[:, j] = (1.0 - df[miss_col].astype(TENSOR_DTYPE).to_numpy()).astype(TENSOR_DTYPE)
        else:
            mask[:, j] = df[col].notna().astype(TENSOR_DTYPE).to_numpy()
    return mask


# -----------------------------------------------------------------------------
# Δt computation — FIXED relative to prior pipeline
# -----------------------------------------------------------------------------
def compute_delta_t(M_window: np.ndarray, bin_hours: float) -> np.ndarray:
    """Compute hours since last observation per feature, per timestep.

    Convention (matches Che et al. 2018):
      - At t=0: Δt = bin_hours (we have not seen prior bins)
      - If observed at t: Δt[t] = bin_hours  (one bin since "this very obs")
      - If missing at t:  Δt[t] = Δt[t-1] + bin_hours

    Shape: M_window is (T, F) -> returns (T, F)

    Critical fix vs prior pipeline: previously Δt was set to 0 when observed,
    making the decay gate's exp(-relu(W*0)) = 1.0 always, defeating the purpose.
    """
    T, F = M_window.shape
    delta = np.zeros((T, F), dtype=TENSOR_DTYPE)
    delta[0, :] = bin_hours  # cold-start assumption
    for t in range(1, T):
        observed_now = M_window[t, :]  # 1 if observed, 0 if missing
        prev_delta = delta[t - 1, :]
        # If observed now: Δt = bin_hours; if missing: Δt = prev + bin_hours
        delta[t, :] = bin_hours + prev_delta * (1.0 - observed_now)
    return delta


def compute_delta_t_batch(M: np.ndarray, bin_hours: float) -> np.ndarray:
    """Vectorised Δt across a batch of windows. M shape (N, T, F)."""
    N, T, F = M.shape
    delta = np.zeros_like(M, dtype=TENSOR_DTYPE)
    delta[:, 0, :] = bin_hours
    for t in range(1, T):
        observed_now = M[:, t, :]
        delta[:, t, :] = bin_hours + delta[:, t - 1, :] * (1.0 - observed_now)
    return delta


# -----------------------------------------------------------------------------
# Sliding-window construction
# -----------------------------------------------------------------------------



def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Map gender {M,F} -> {0.0,1.0}. Idempotent.

    Detection is content-based, not dtype-based, because pandas may store
    short string columns under either `object` or `str` (pandas 2.x string
    dtype). If the column is already numeric, it is left untouched.
    """
    df = df.copy()
    col = df["gender"]
    # If already numeric, do nothing (idempotency).
    if pd.api.types.is_numeric_dtype(col):
        df["gender"] = col.astype(TENSOR_DTYPE)
        return df
    # Otherwise treat as string-like and map.
    df["gender"] = col.map({"M": 0.0, "F": 1.0}).astype(TENSOR_DTYPE)
    return df


def censor_post_onset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bins occurring AFTER the patient's sepsis onset.

    The model must not be allowed to see post-onset trajectory; that would
    leak the label.
    """
    is_pos = df["sepsis3_onset_time"].notnull()
    valid = df["bin_end"] <= df["sepsis3_onset_time"]
    keep = (~is_pos) | (is_pos & valid)
    return df.loc[keep].copy()


def add_time_to_onset(df: pd.DataFrame, sentinel: float = 999.0) -> pd.DataFrame:
    """Add `time_to_onset` (hours from bin_end to onset). Negatives censored."""
    df = df.copy()
    tto = (df["sepsis3_onset_time"] - df["bin_end"]).dt.total_seconds() / 3600.0
    tto = tto.fillna(sentinel).astype(TENSOR_DTYPE)
    df["time_to_onset"] = tto
    return df


def _count_windows_per_split(df: pd.DataFrame, lookback: int) -> Dict[str, int]:
    """First pass: count exact windows per split so we can pre-allocate."""
    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    sizes = df.groupby("stay_id").size()
    splits = df.groupby("stay_id")["split"].first()
    for stay_id, n in sizes.items():
        if n < lookback:
            continue
        counts[splits[stay_id]] += (n - lookback + 1)
    return counts


def build_and_save_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    bin_hours: float,
    out_dir: Path,
) -> Dict[str, int]:
    """Build sliding windows and write per-split tensors to disk incrementally.

    Memory-efficient: pre-allocates one output array per split using the
    counted size, then fills in place. Avoids holding all windows in Python
    lists (which is ~3-4x the final memory).

    Writes:
      X_{split}.npy, M_{split}.npy, D_{split}.npy, y_{split}.npy,
      {split}_row_ids.npy

    Returns dict of split -> count written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["stay_id", "bin_index"]).reset_index(drop=True)
    F = len(feature_cols)

    # Pass 1: count windows per split for exact pre-allocation
    counts = _count_windows_per_split(df, lookback)

    # Pre-allocate per-split arrays
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    cursors: Dict[str, int] = {s: 0 for s in counts}
    for split_name, n in counts.items():
        if n == 0:
            continue
        arrays[split_name] = {
            "X": np.empty((n, lookback, F), dtype=TENSOR_DTYPE),
            "M": np.empty((n, lookback, F), dtype=TENSOR_DTYPE),
            "D": np.empty((n, lookback, F), dtype=TENSOR_DTYPE),
            "y": np.empty((n, 2), dtype=TENSOR_DTYPE),
            "ids": np.empty(n, dtype=object),
        }

    # Pass 2: fill arrays per stay
    for stay_id, grp in df.groupby("stay_id", sort=False):
        if len(grp) < lookback:
            continue
        X_full = grp[feature_cols].to_numpy(dtype=TENSOR_DTYPE)
        M_full = build_observation_mask(grp, feature_cols)
        D_full = compute_delta_t(M_full, bin_hours)
        y_label = grp["sepsis_next_6h"].to_numpy(dtype=TENSOR_DTYPE)
        y_tto = grp["time_to_onset"].to_numpy(dtype=TENSOR_DTYPE)
        bin_idx_arr = grp["bin_index"].to_numpy()
        split_val = grp["split"].iloc[0]
        subj = int(grp["subject_id"].iloc[0])
        sid = int(stay_id)
        store = arrays[split_val]

        for end in range(lookback - 1, len(grp)):
            start = end - lookback + 1
            c = cursors[split_val]
            store["X"][c] = X_full[start:end + 1]
            store["M"][c] = M_full[start:end + 1]
            store["D"][c] = D_full[start:end + 1]
            store["y"][c, 0] = y_label[end]
            store["y"][c, 1] = y_tto[end]
            store["ids"][c] = f"{subj}_{sid}_{int(bin_idx_arr[end])}"
            cursors[split_val] = c + 1

    # Verify cursors match counts (defensive)
    for split_name, n in counts.items():
        if n > 0 and cursors[split_name] != n:
            raise RuntimeError(
                f"Window count mismatch for split={split_name}: "
                f"expected {n}, filled {cursors[split_name]}"
            )

    # Write to disk
    for split_name, store in arrays.items():
        np.save(out_dir / f"X_{split_name}.npy", store["X"])
        np.save(out_dir / f"M_{split_name}.npy", store["M"])
        np.save(out_dir / f"D_{split_name}.npy", store["D"])
        np.save(out_dir / f"y_{split_name}.npy", store["y"])
        # Convert ids to fixed-width string array for npy
        ids_str = store["ids"].astype(str)
        np.save(out_dir / f"{split_name}_row_ids.npy", ids_str)

    return counts


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------
def tensorize(cfg: Dict, parquet_path: Path, out_dir: Path) -> Dict:
    """End-to-end tensorization. Returns a summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    df["bin_start"] = pd.to_datetime(df["bin_start"])
    df["bin_end"] = pd.to_datetime(df["bin_end"])
    df["sepsis3_onset_time"] = pd.to_datetime(df["sepsis3_onset_time"])

    df = encode_gender(df)
    df = censor_post_onset(df)
    df = add_time_to_onset(df)

    feature_cols = get_feature_columns(cfg)
    binary_cols = [c for c in get_binary_columns(cfg) if c in feature_cols]
    scale_cols = [c for c in feature_cols if c not in binary_cols]

    train_mask = (df["split"] == "train")
    scaler = fit_robust_scaler(df, train_mask, scale_cols)
    df = apply_robust_scaler(df, scale_cols, scaler)

    # Final imputation: scaled NaNs -> 0.0 (== train median in scaled space)
    df[scale_cols] = df[scale_cols].fillna(0.0).astype(TENSOR_DTYPE)
    df[binary_cols] = df[binary_cols].fillna(0.0).astype(TENSOR_DTYPE)

    counts = build_and_save_windows(
        df,
        feature_cols=feature_cols,
        lookback=cfg["cohort"]["lookback_bins"],
        bin_hours=cfg["cohort"]["bin_hours"],
        out_dir=out_dir,
    )

    summary: Dict = {"feature_cols": feature_cols, "n_features": len(feature_cols)}
    for split_name in ["train", "val", "test"]:
        if counts.get(split_name, 0) == 0:
            continue
        y_split = np.load(out_dir / f"y_{split_name}.npy", mmap_mode="r")
        summary[f"n_{split_name}"] = int(counts[split_name])
        summary[f"prevalence_{split_name}"] = float(y_split[:, 0].mean())

    # Persist scaling params and feature names
    np.savez(
        out_dir / "scaling_params.npz",
        median=scaler["median"], iqr=scaler["iqr"], columns=scaler["columns"],
    )
    (out_dir / "feature_names.txt").write_text("\n".join(feature_cols))

    # Hash the test row ids — used by downstream notebooks to verify alignment.
    test_ids = np.load(out_dir / "test_row_ids.npy")
    summary["test_row_id_hash"] = hashlib.md5(test_ids.tobytes()).hexdigest()

    with open(out_dir / "cohort_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
