"""
Model architectures (PyTorch).

This module contains all neural baselines and the proposed TD-GRU.

Critical design decisions vs the prior pipeline
-----------------------------------------------

1. **Faithful GRU-D** (Che et al. 2018, Sci. Rep. 8:6085).
   - Decay applied to BOTH input features and hidden state.
   - Inputs use the empirical mean fallback (in scaled space, this is 0).
   - Concatenates the observation MASK as an explicit input channel.
   - Initialises decay weights with a small positive value so gammas are
     non-trivial from epoch 1 (zero-init was a real bug previously).

2. **TD-GRU (proposed)**.
   - Decay applied to HIDDEN STATE ONLY. Raw inputs preserved (not decayed)
     so acute physiological abnormalities are not artificially smoothed.
   - Decay weight matrix maps Δt from feature-space to hidden-space.
   - Mask is concatenated as input (so the model knows what was observed).

3. **All RNN cells consume the same input format**: concatenation
   [X (F), M (F)] of shape (B, T, 2F). Δt is passed as a SEPARATE tensor
   to the forward call rather than concatenated. This makes the
   feature-vs-Δt distinction explicit in code.

4. **Standard GRU baseline** sees [X, M] as inputs but no Δt at all.
   This is the cleanest possible "no time-awareness" control.

5. **Standard GRU + Δt-as-feature** baseline (for honest ablation):
   sees [X, M, Δt] as plain features. This isolates the contribution
   of "Δt as decay driver" vs "Δt as just another feature".
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# =============================================================================
# Custom recurrent cells
# =============================================================================
class GRUDCell(nn.Module):
    """Faithful GRU-D cell (Che et al. 2018).

    Input per timestep: x_t (F,), m_t (F,), delta_t (F,).
    Decay weights are POSITIVE (enforced via softplus) and applied as
    gamma_x = exp(-relu(W_x . delta_t))  -> per-feature scalar via diag
    gamma_h = exp(-relu(W_h . delta_t))  -> hidden-dim vector

    Imputed input:
        x_hat_t = m_t * x_t + (1 - m_t) * (gamma_x * x_prev + (1 - gamma_x) * x_mean)
    where x_mean=0 in scaled space and x_prev is the last observed value.

    To keep the implementation tractable inside a vectorised RNN loop, we
    track x_prev as part of the cell state.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.F = input_dim
        self.H = hidden_dim

        # Per-feature decay for inputs (diagonal); positive via softplus
        self.w_dec_x = nn.Parameter(torch.zeros(input_dim) + 0.01)
        # Hidden-state decay: project Δt (F,) -> (H,)
        self.w_dec_h = nn.Linear(input_dim, hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.w_dec_h.weight, gain=0.1)
        nn.init.zeros_(self.w_dec_h.bias)

        # GRU cell consumes [x_hat, m] of size 2F
        self.gru = nn.GRUCell(input_dim * 2, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.H, device=device)
        x_prev = torch.zeros(batch_size, self.F, device=device)
        return h, x_prev

    def forward(
        self,
        x: torch.Tensor,    # (B, F)
        m: torch.Tensor,    # (B, F)
        d: torch.Tensor,    # (B, F)
        h: torch.Tensor,    # (B, H)
        x_prev: torch.Tensor,  # (B, F)  last-observed (or 0)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-feature gamma_x in (0, 1]
        gamma_x = torch.exp(-torch.relu(self.w_dec_x.unsqueeze(0) * d))  # (B, F)
        # Hidden gamma in (0, 1]^H
        gamma_h = torch.exp(-torch.relu(self.w_dec_h(d)))  # (B, H)

        # Imputed inputs: observed -> x; missing -> decayed last-obs (mean=0 in z-space)
        x_imp_missing = gamma_x * x_prev  # (B, F); fallback toward 0 (mean)
        x_hat = m * x + (1.0 - m) * x_imp_missing

        # Decayed hidden
        h_dec = h * gamma_h
        h_new = self.gru(torch.cat([x_hat, m], dim=-1), h_dec)

        # Update last-observed
        x_prev_new = m * x + (1.0 - m) * x_prev
        return h_new, x_prev_new


class TDGRUCell(nn.Module):
    """Time-Decay GRU (proposed): decay hidden state ONLY; preserve raw inputs.

    Input: [x_t, m_t] of size 2F passed to inner GRUCell.
    Δt is consumed only by the decay gate that scales the previous hidden state.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.F = input_dim
        self.H = hidden_dim
        # Δt -> hidden-dim gate
        self.w_dec_h = nn.Linear(input_dim, hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.w_dec_h.weight, gain=0.1)
        nn.init.zeros_(self.w_dec_h.bias)
        self.gru = nn.GRUCell(input_dim * 2, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.H, device=device)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        d: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        gamma_h = torch.exp(-torch.relu(self.w_dec_h(d)))  # (B, H), in (0, 1]
        h_dec = h * gamma_h
        # Note: x is preserved, NOT decayed (key TD-GRU vs GRU-D distinction)
        h_new = self.gru(torch.cat([x, m], dim=-1), h_dec)
        return h_new


# =============================================================================
# Sequence wrappers
# =============================================================================
class TDGRU(nn.Module):
    """Sequence wrapper around TDGRUCell with classification head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(input_dim * 2)
        self.cell = TDGRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # x, m, d each (B, T, F)
        B, T, F = x.shape
        # LayerNorm applied to the [x, m] concatenation per timestep
        cat = self.layer_norm(torch.cat([x, m], dim=-1))  # (B, T, 2F)
        x_n = cat[..., :F]
        m_n = cat[..., F:]
        h = self.cell.init_state(B, x.device)
        for t in range(T):
            h = self.cell(x_n[:, t], m_n[:, t], d[:, t], h)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)  # (B,) logits


class GRUD(nn.Module):
    """Sequence wrapper around faithful GRUDCell."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.cell = GRUDCell(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        cat = self.layer_norm(torch.cat([x, m], dim=-1))
        x_n = cat[..., :F]
        m_n = cat[..., F:]
        h, x_prev = self.cell.init_state(B, x.device)
        for t in range(T):
            h, x_prev = self.cell(x_n[:, t], m_n[:, t], d[:, t], h, x_prev)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


class StandardGRU(nn.Module):
    """Plain GRU baseline; sees [X, M] but NOT Δt. Cleanest 'no time-aware' control."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim * 2)
        self.gru = nn.GRU(input_dim * 2, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor = None) -> torch.Tensor:
        cat = self.layer_norm(torch.cat([x, m], dim=-1))
        out, h = self.gru(cat)
        h = h.squeeze(0)  # (B, H)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


class StandardGRUWithDeltaFeatures(nn.Module):
    """Standard GRU that gets Δt concatenated as plain features.

    This is the *honest* ablation that disentangles "Δt as decay driver"
    (in TD-GRU) from "Δt as feature" (here).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        in_total = input_dim * 3  # X, M, D
        self.layer_norm = nn.LayerNorm(in_total)
        self.gru = nn.GRU(in_total, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        cat = self.layer_norm(torch.cat([x, m, d], dim=-1))
        out, h = self.gru(cat)
        h = h.squeeze(0)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


class CNN1D(nn.Module):
    """1D-CNN baseline. Sees [X, M] as channels (no Δt)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        in_ch = input_dim * 2  # X and M concatenated as channels
        self.conv1 = nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor = None) -> torch.Tensor:
        # PyTorch Conv1d expects (B, C, T); permute from (B, T, F)
        cat = torch.cat([x, m], dim=-1).transpose(1, 2)  # (B, 2F, T)
        h = self.act(self.conv1(cat))
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        h = self.act(self.fc(h))
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


class TransformerEncoder(nn.Module):
    """Multi-head attention transformer with continuous-time positional encoding.

    Δt is encoded into the positional embedding via a small MLP, so the
    Transformer is informed about elapsed time (a proper time-aware baseline).
    Without this, a Transformer on T=8 with positional encodings is essentially
    blind to irregular intervals.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        in_total = input_dim * 2
        self.input_proj = nn.Linear(in_total, hidden_dim)
        # Continuous-time PE: Δt vector -> hidden_dim
        self.time_pe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x, m], dim=-1)         # (B, T, 2F)
        h = self.input_proj(cat)                # (B, T, H)
        pe = self.time_pe(d)                    # (B, T, H)
        h = h + pe
        h = self.encoder(h)
        # Pool over time
        h = h.mean(dim=1)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


# =============================================================================
# Factory
# =============================================================================
def _init_head_bias(model: nn.Module, prevalence: float) -> nn.Module:
    """Initialize the output head's bias to log-odds of the prevalence.

    This ensures the model starts predicting near the base rate (~0.5%)
    rather than 50%, dramatically improving early-epoch calibration and
    gradient stability under class-weighted losses.
    """
    import math
    bias_val = math.log(prevalence / (1.0 - prevalence + 1e-9))
    if hasattr(model, "head") and hasattr(model.head, "bias") and model.head.bias is not None:
        nn.init.constant_(model.head.bias, bias_val)
    return model


def build_model(name: str, input_dim: int, cfg: dict, prevalence: float = 0.0054) -> nn.Module:
    """Instantiate a model by short name with prevalence-aware bias init."""
    H = cfg["training"]["hidden_size"]
    dr = cfg["training"]["dropout"]
    name = name.lower()
    if name == "td_gru":
        m = TDGRU(input_dim, H, dr)
    elif name == "grud":
        m = GRUD(input_dim, H, dr)
    elif name == "gru":
        m = StandardGRU(input_dim, H, dr)
    elif name == "gru_delta_features":
        m = StandardGRUWithDeltaFeatures(input_dim, H, dr)
    elif name == "cnn1d":
        m = CNN1D(input_dim, H, dr)
    elif name == "transformer":
        m = TransformerEncoder(input_dim, H, n_heads=4, n_blocks=2, dropout=dr)
    else:
        raise ValueError(f"Unknown model: {name}")
    return _init_head_bias(m, prevalence)
