"""
Statistical evaluation utilities.

Includes
--------
1. `paired_bootstrap_pr_auc_test` — proper paired bootstrap test for PR-AUC
   differences. n>=2000, two-sided p-value, BCa-style symmetric quantile CI.
2. `bootstrap_ci` — single-model bootstrap confidence interval.
3. `calibrate_isotonic_on_val` — fit isotonic on val, apply to test.
   The decision to use raw vs calibrated MUST be made on val PR-AUC,
   never on test (this was a real issue in the prior pipeline).
4. `physionet2019_utility` — official PhysioNet 2019 sepsis utility score.
   Adapted for our 6h fixed-horizon task.
5. `decision_curve_analysis` — Vickers-Elkin net benefit.
6. `clinical_utility_at_sensitivity` — operational metrics at a target Sn.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


# -----------------------------------------------------------------------------
# Bootstrap statistics
# -----------------------------------------------------------------------------
def _safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "pr_auc",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    random_seed: int = 12345,
) -> Dict[str, float]:
    """Bootstrap CI for a single model's metric.

    metric in {"pr_auc", "roc_auc", "brier"}.
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    vals = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if metric == "pr_auc":
            vals[i] = _safe_pr_auc(y_true[idx], y_prob[idx])
        elif metric == "roc_auc":
            try:
                vals[i] = roc_auc_score(y_true[idx], y_prob[idx])
            except ValueError:
                vals[i] = float("nan")
        elif metric == "brier":
            vals[i] = brier_score_loss(y_true[idx], y_prob[idx])
        else:
            raise ValueError(metric)
    vals = vals[~np.isnan(vals)]
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    point = (
        _safe_pr_auc(y_true, y_prob) if metric == "pr_auc"
        else roc_auc_score(y_true, y_prob) if metric == "roc_auc"
        else brier_score_loss(y_true, y_prob)
    )
    return {"point": float(point), "ci_low": lo, "ci_high": hi, "n_resamples": int(n_resamples)}


def paired_bootstrap_pr_auc_test(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_resamples: int = 2000,
    random_seed: int = 12345,
) -> Dict[str, float]:
    """Paired bootstrap test for difference in PR-AUC (model A - model B).

    The same resampled indices are used for both models, so the test is
    properly paired and accounts for shared data variance.

    Returns a two-sided p-value derived from the bootstrap distribution
    centered under the null (mean-shift the diff distribution to zero
    and report the fraction at least as extreme as the observed diff).
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)

    obs_diff = _safe_pr_auc(y_true, p_a) - _safe_pr_auc(y_true, p_b)
    diffs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ya = y_true[idx]
        diffs[i] = _safe_pr_auc(ya, p_a[idx]) - _safe_pr_auc(ya, p_b[idx])

    diffs = diffs[~np.isnan(diffs)]

    # Two-sided p via centring at zero: Bp = diffs - mean(diffs)
    centred = diffs - diffs.mean()
    p_two = float(np.mean(np.abs(centred) >= abs(obs_diff)))
    # Floor at 1/n_resamples + 1 to avoid reporting "p = 0"
    p_two = max(p_two, 1.0 / (n_resamples + 1))

    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return {
        "obs_diff": float(obs_diff),
        "p_value_two_sided": float(p_two),
        "ci_low": lo,
        "ci_high": hi,
        "n_resamples": int(n_resamples),
    }


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
def calibrate_isotonic_on_val(
    val_probs: np.ndarray, val_labels: np.ndarray,
    test_probs: np.ndarray, test_labels: np.ndarray,
    val_threshold_improvement: float = 0.001,
) -> Tuple[np.ndarray, str]:
    """Fit isotonic on VAL, apply to TEST.

    Returns the test probabilities to use AND a string flag
    ("calibrated" or "raw") indicating which version was selected.

    The selection rule is purely VAL-based: if calibration improves
    val PR-AUC by at least `val_threshold_improvement`, return the
    calibrated test predictions. Otherwise return raw.

    This fixes the prior pipeline's bug of choosing raw vs calibrated
    based on a TEST-set difference.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(val_probs, val_labels)
    val_cal = iso.predict(val_probs)
    val_pr_raw = average_precision_score(val_labels, val_probs)
    val_pr_cal = average_precision_score(val_labels, val_cal)

    if val_pr_cal > val_pr_raw + val_threshold_improvement:
        return iso.predict(test_probs), "calibrated"
    return test_probs, "raw"


def reliability_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


# -----------------------------------------------------------------------------
# Clinical utility
# -----------------------------------------------------------------------------
def clinical_utility_at_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float = 0.80,
) -> Dict[str, float]:
    """Find the threshold that achieves >= target sensitivity, report metrics."""
    # Sort by prob descending; sweep
    order = np.argsort(-y_prob)
    y_sorted = y_true[order].astype(int)
    p_sorted = y_prob[order]

    cum_tp = np.cumsum(y_sorted)
    total_pos = y_sorted.sum()
    sensitivities = cum_tp / max(1, total_pos)
    # Smallest k with sensitivity >= target
    above = np.where(sensitivities >= target_sensitivity)[0]
    if len(above) == 0:
        return {"threshold": float("nan"), "achieved_sensitivity": 0.0}
    k = int(above[0])
    threshold = float(p_sorted[k])

    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    ppv = tp / max(1, (tp + fp))
    wdr = (tp + fp) / max(1, tp)
    nns = len(y_true) / max(1, tp)
    return {
        "threshold": threshold,
        "achieved_sensitivity": float(sens),
        "specificity": float(spec),
        "precision_ppv": float(ppv),
        "wdr": float(wdr),
        "nns": float(nns),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def decision_curve_analysis(
    y_true: np.ndarray, y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Vickers-Elkin net benefit at each threshold.

    NB(t) = TP/N - FP/N * t/(1-t)
    """
    n = len(y_true)
    nb = np.empty_like(thresholds, dtype=np.float64)
    for i, t in enumerate(thresholds):
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        nb[i] = tp / n - (fp / n) * (t / (1.0 - t + 1e-12))
    return nb


# -----------------------------------------------------------------------------
# PhysioNet 2019 sepsis utility (adapted)
# -----------------------------------------------------------------------------
def physionet2019_utility(
    y_true_per_window: np.ndarray,
    y_pred_binary_per_window: np.ndarray,
    time_to_onset_per_window: np.ndarray,
    dt_early: float = 12.0,
    dt_optimal: float = 6.0,
    dt_late: float = 3.0,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
) -> float:
    """Per-window adaptation of the PhysioNet 2019 sepsis utility metric.

    Reyna et al. (Critical Care Medicine, 2020) define a per-hour scoring
    that rewards predictions made between (t_onset - dt_early, t_onset),
    peaking at t_onset - dt_optimal. We adapt to per-window scoring by using
    the window's `time_to_onset` (hours from window-end to onset) for positives.

    For negatives, FP carries `u_fp` cost, TN gives `u_tn`.

    Returns a normalized utility in [0, 1] where:
      - perfect = always optimal-time positive prediction on positives, no FP
      - inaction = predict 0 on everything
    Final score = (achieved - inaction) / (perfect - inaction).
    """
    y_true = np.asarray(y_true_per_window).astype(int)
    y_pred = np.asarray(y_pred_binary_per_window).astype(int)
    tto = np.asarray(time_to_onset_per_window).astype(float)

    def per_window_reward(label: int, pred: int, t: float) -> float:
        if label == 1:
            if pred == 1:
                # Reward triangle: 0 at t=dt_early, max at t=dt_optimal,
                # decays to min_u_fn at t=-dt_late (i.e., late prediction)
                if t > dt_early:
                    return 0.0
                if t >= dt_optimal:
                    # Linear from 0 at dt_early down to max_u_tp at dt_optimal
                    return max_u_tp * (dt_early - t) / max(dt_early - dt_optimal, 1e-9)
                if t >= 0:
                    # Linear from max_u_tp at dt_optimal to ~0 at onset
                    return max_u_tp * (t / max(dt_optimal, 1e-9))
                # t < 0 is post-onset: penalise as late
                if t >= -dt_late:
                    return min_u_fn * (-t / max(dt_late, 1e-9))
                return min_u_fn
            else:  # missed positive
                return min_u_fn
        else:  # negative
            return u_fp if pred == 1 else u_tn

    achieved = 0.0
    perfect = 0.0
    inaction = 0.0
    for label, pred, t in zip(y_true, y_pred, tto):
        achieved += per_window_reward(label, pred, t)
        # Best possible: positives -> max_u_tp at exactly dt_optimal; negatives -> 0
        perfect += max_u_tp if label == 1 else 0.0
        inaction += min_u_fn if label == 1 else u_tn  # all-zero predictor

    denom = perfect - inaction
    if denom == 0:
        return 0.0
    return float((achieved - inaction) / denom)


def physionet2019_utility_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, tto: np.ndarray, threshold: float,
) -> float:
    return physionet2019_utility(y_true, (y_prob >= threshold).astype(int), tto)


def find_best_utility_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, tto: np.ndarray,
    n_thresholds: int = 50,
) -> Tuple[float, float]:
    """Sweep threshold quantiles and return (best_threshold, best_utility)."""
    qs = np.linspace(0.5, 0.999, n_thresholds)
    cands = np.quantile(y_prob, qs)
    best_t, best_u = float("nan"), -float("inf")
    for t in cands:
        u = physionet2019_utility_at_threshold(y_true, y_prob, tto, t)
        if u > best_u:
            best_u, best_t = u, float(t)
    return best_t, best_u
