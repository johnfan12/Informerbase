from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_3d(tensor):
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected tensor with shape [batch, length, channels], got {tensor.shape}"
        )


def _moving_average(sequence: torch.Tensor, window_size: int) -> torch.Tensor:
    """Compute a simple moving average along the temporal axis."""
    _ensure_3d(sequence)
    if window_size <= 1:
        return sequence

    batch, length, channels = sequence.shape
    # reshape so conv1d can treat each feature independently
    series = sequence.permute(0, 2, 1).reshape(batch * channels, 1, length)

    left_pad = (window_size - 1) // 2
    right_pad = window_size - 1 - left_pad
    kernel = torch.ones(1, 1, window_size, device=sequence.device, dtype=sequence.dtype)
    kernel = kernel / window_size

    padded = F.pad(series, (left_pad, right_pad), mode='replicate')
    trend = F.conv1d(padded, kernel)
    trend = trend.reshape(batch, channels, length).permute(0, 2, 1)
    return trend


def _trend_loss(y_true: torch.Tensor, y_pred: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return torch.zeros(1, device=y_true.device, dtype=y_true.dtype)
    true_trend = _moving_average(y_true, window)
    pred_trend = _moving_average(y_pred, window)
    return F.mse_loss(pred_trend, true_trend)


def _season_loss(y_true: torch.Tensor, y_pred: torch.Tensor, period: Optional[int]) -> torch.Tensor:
    if period is None or period < 2:
        return torch.zeros(1, device=y_true.device, dtype=y_true.dtype)

    _ensure_3d(y_true)
    batch, length, channels = y_true.shape
    trimmed = (length // period) * period
    if trimmed == 0:
        return torch.zeros(1, device=y_true.device, dtype=y_true.dtype)

    true_cycles = y_true[:, :trimmed, :].reshape(batch, -1, period, channels)
    pred_cycles = y_pred[:, :trimmed, :].reshape(batch, -1, period, channels)

    true_pattern = true_cycles.mean(dim=1)
    pred_pattern = pred_cycles.mean(dim=1)
    return F.mse_loss(pred_pattern, true_pattern)


def _find_jump_indices(
    y_true: torch.Tensor, horizon: int, threshold: float
) -> torch.Tensor:
    _ensure_3d(y_true)
    batch, length, channels = y_true.shape
    if horizon <= 0 or length < 2:
        return torch.zeros((batch, length), device=y_true.device, dtype=torch.bool)

    jump_mask = torch.zeros((batch, length), device=y_true.device, dtype=torch.bool)
    for step in range(1, horizon + 1):
        if step >= length:
            break
        delta = torch.abs(y_true[:, step:, :] - y_true[:, :-step, :])
        delta = delta.max(dim=-1).values
        pad = torch.zeros((batch, step), device=y_true.device, dtype=delta.dtype)
        delta = torch.cat([pad, delta], dim=1)
        jump_mask = jump_mask | (delta > threshold)
    return jump_mask


def _build_pre_jump_mask(jump_mask: torch.Tensor, pre_days: int) -> torch.Tensor:
    batch, length = jump_mask.shape
    if pre_days <= 0:
        return jump_mask.float()

    weight = jump_mask.float()
    max_offset = min(pre_days, max(0, length - 1))
    for offset in range(1, max_offset + 1):
        pad = torch.zeros((batch, offset), device=jump_mask.device, dtype=jump_mask.dtype)
        shifted = torch.cat([jump_mask[:, offset:], pad], dim=1)
        weight += shifted.float()
    return weight


def _jump_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    horizon: int,
    threshold: float,
    pre_days: int,
    base_weight: float,
    jump_scale: float,
) -> torch.Tensor:
    jump_mask = _find_jump_indices(y_true, horizon, threshold)
    pre_mask = _build_pre_jump_mask(jump_mask, pre_days)
    weights = base_weight + jump_scale * pre_mask
    if torch.all(weights <= 0):
        return torch.zeros(1, device=y_true.device, dtype=y_true.dtype)
    weights = weights.clamp(min=1e-6).unsqueeze(-1)

    error = (y_pred - y_true) ** 2
    weighted_error = error * weights
    normalization = weights.sum().clamp(min=1.0)
    return weighted_error.sum() / normalization


class MultiObjectiveTimeSeriesLoss(nn.Module):
    """Weighted combination of point-wise, trend, seasonal, and jump-sensitive losses."""

    def __init__(
        self,
        *,
        period: Optional[int] = None,
        trend_window: int = 7,
        jump_horizon: int = 2,
        jump_threshold: float = 1.0,
        jump_pre_days: int = 5,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        delta: float = 1.0,
        jump_base_weight: float = 1.0,
        jump_scale: float = 5.0,
        return_components: bool = False,
    ) -> None:
        super().__init__()
        self.period = period if period and period > 0 else None
        self.trend_window = max(1, trend_window)
        self.jump_horizon = max(0, jump_horizon)
        self.jump_threshold = jump_threshold
        self.jump_pre_days = max(0, jump_pre_days)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.jump_base_weight = jump_base_weight
        self.jump_scale = jump_scale
        self.return_components = return_components

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Predictions and targets must share the same shape, got {y_pred.shape} vs {y_true.shape}"
            )
        _ensure_3d(y_true)

        mse = F.mse_loss(y_pred, y_true)
        zero = torch.zeros(1, device=y_true.device, dtype=y_true.dtype)
        trend = (
            _trend_loss(y_true, y_pred, self.trend_window)
            if self.beta != 0
            else zero
        )
        season = (
            _season_loss(y_true, y_pred, self.period)
            if self.gamma != 0
            else zero
        )
        jump = (
            _jump_loss(
                y_true,
                y_pred,
                self.jump_horizon,
                self.jump_threshold,
                self.jump_pre_days,
                self.jump_base_weight,
                self.jump_scale,
            )
            if self.delta != 0
            else zero
        )

        total = self.alpha * mse + self.beta * trend + self.gamma * season + self.delta * jump

        if self.return_components:
            components = {
                'mse': mse.detach(),
                'trend': trend.detach(),
                'season': season.detach(),
                'jump': jump.detach(),
                'total': total.detach(),
            }
            return total, components
        return total
