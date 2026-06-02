"""
Loss functions for highly imbalanced sepsis prediction.

Two losses are exposed:

1. `WeightedBCELoss` — standard binary cross-entropy with positive class
   weight. Used by the "no ATP" ablation and by simple baselines.

2. `AsymmetricTemporalPenaltyLoss` (ATP) — combines focal modulation,
   class weighting, and a temporal penalty that up-weights positives
   with imminent onset (small time-to-onset).

Important design note on pos_weight
------------------------------------
Using the full inverse-prevalence ratio (N_neg/N_pos ~ 191) produces
extreme gradients that destroy probability calibration (Brier scores
explode to 0.2+). The correct approach for focal-style losses under
extreme imbalance is to use a DAMPED weight:

  - sqrt(N_neg / N_pos)  — recommended, ~14 at 0.5% prevalence
  - log(N_neg / N_pos)   — alternative, ~5.25

This gives the model enough positive gradient to learn the minority
class without forcing systematic over-prediction. The `compute_pos_weight`
utility function below handles this.

The caller is responsible for computing pos_weight on TRAIN split only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(y_train_labels, method: str = "sqrt") -> float:
    """Compute a calibration-safe positive class weight.

    Parameters
    ----------
    y_train_labels : array-like of {0, 1}
        Binary labels from the training split.
    method : str
        'sqrt'  — sqrt(N_neg / N_pos).  Recommended.
        'log'   — log(N_neg / N_pos).
        'full'  — N_neg / N_pos.  WARNING: destroys calibration at low prevalence.
        'none'  — returns 1.0  (unweighted).

    Returns
    -------
    float
    """
    import numpy as np
    y = np.asarray(y_train_labels).ravel()
    n_pos = max(int(y.sum()), 1)
    n_neg = len(y) - n_pos
    ratio = float(n_neg) / float(n_pos)
    if method == "sqrt":
        return math.sqrt(ratio)
    if method == "log":
        return math.log(max(ratio, 1.0))
    if method == "full":
        return ratio
    if method == "none":
        return 1.0
    raise ValueError(f"Unknown method: {method}")


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with a constant positive-class weight."""

    def __init__(self, pos_weight: float):
        super().__init__()
        self.register_buffer("pos_weight_buf", torch.tensor(pos_weight, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (B,) raw scores. target: (B, 2) [label, time_to_onset]."""
        y = target[:, 0].float()
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y, pos_weight=self.pos_weight_buf, reduction="mean"
        )
        return loss


class AsymmetricTemporalPenaltyLoss(nn.Module):
    """ATP loss as described in the manuscript.

    Parameters
    ----------
    pos_weight : float
        Damped positive weight (use compute_pos_weight with method='sqrt').
    focal_gamma : float, default 1.5
        Focal modulation exponent. 0 disables focal.
    temporal_decay_alpha : float, default 0.5
        Temporal urgency rate. 0 disables temporal penalty.
    horizon_hours : float, default 6.0
        Boundary defining "imminent" cases.
    cap_temporal_penalty : float, default 20.0
        Numerical stability: clamp the temporal multiplier.
    """

    def __init__(
        self,
        pos_weight: float,
        focal_gamma: float = 1.5,
        temporal_decay_alpha: float = 0.5,
        horizon_hours: float = 6.0,
        cap_temporal_penalty: float = 20.0,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.focal_gamma = float(focal_gamma)
        self.alpha = float(temporal_decay_alpha)
        self.H = float(horizon_hours)
        self.cap = float(cap_temporal_penalty)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = target[:, 0].float()
        tto = target[:, 1].float()

        logits = logits.view(-1)
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, eps, 1.0 - eps)

        # Per-sample BCE (no reduction yet)
        bce = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))

        # Focal modulation on p_t = p if y=1 else 1-p
        p_t = y * p + (1.0 - y) * (1.0 - p)
        focal_mod = (1.0 - p_t).pow(self.focal_gamma)

        # Class weight (only positive samples up-weighted)
        cls_w = torch.where(
            y > 0.5,
            torch.full_like(y, self.pos_weight),
            torch.ones_like(y),
        )

        # Temporal penalty: only active for positives with tto in [0, H].
        urgency = torch.clamp(self.H - tto, min=0.0)
        temp_pen_pos = torch.exp(self.alpha * urgency).clamp(max=self.cap)
        temp_pen = torch.where(y > 0.5, temp_pen_pos, torch.ones_like(y))

        loss = focal_mod * cls_w * temp_pen * bce
        return loss.mean()
