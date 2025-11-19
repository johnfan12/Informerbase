import re
import textwrap
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None


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


class LLMScoreLoss(nn.Module):
    """Leverages an instruction-tuned LLM (e.g., Qwen) to score predictions.

    The model receives a prompt containing the historical look-back window,
    the model's prediction, and the ground-truth values. It must respond with a
    numeric score inside ``<score>...</score>`` tags. The reciprocal of that
    score becomes the training loss. This loss is *not differentiable* with
    respect to the forecaster's parameters, so it should be combined with
    differentiable losses or used for monitoring.
    """

    requires_context = True

    _SCORE_PATTERN = re.compile(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>", re.IGNORECASE)

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        min_score: float = 1.0,
        fallback_score: float = 10.0,
        max_rows: int = 32,
        precision: int = 4,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for LLMScoreLoss. Install it via pip install transformers."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        self.primary_device = self._infer_primary_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.min_score = max(1e-6, min_score)
        self.fallback_score = max(self.min_score, fallback_score)
        self.max_rows = max_rows
        self.precision = precision
        self.system_prompt = system_prompt or (
            "You are a strict time-series auditor. Rate how well a prediction "
            "matches the ground truth on a 0-100 scale where 100 means perfect." 
            " Provide <score>number</score> only after a short justification."
        )
        self.user_prompt_template = user_prompt_template or textwrap.dedent(
            """
            Review the forecast below.

            Look-back sequence:
            {lookback}

            Model prediction:
            {prediction}

            Ground truth:
            {truth}

            Output a short critique explaining the main discrepancies. Then provide
            a holistic score between 0 and 100 inside <score></score>. Higher scores
            indicate closer alignment to the ground truth.
            """
        ).strip()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        *,
        lookback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must share the same shape for LLM scoring")

        y_pred_cpu = y_pred.detach().cpu()
        y_true_cpu = y_true.detach().cpu()
        lookback_cpu = lookback.detach().cpu() if lookback is not None else None

        batch_losses = []
        for idx in range(y_pred_cpu.shape[0]):
            prompt = self._build_prompt(
                lookback_cpu[idx] if lookback_cpu is not None else None,
                y_pred_cpu[idx],
                y_true_cpu[idx],
            )
            score = self._query_llm(prompt)
            batch_losses.append(self._score_to_loss(score))

        loss_tensor = torch.tensor(batch_losses, dtype=y_pred.dtype, device=y_pred.device)
        return loss_tensor.mean()

    def _build_prompt(
        self,
        lookback: Optional[torch.Tensor],
        prediction: torch.Tensor,
        truth: torch.Tensor,
    ) -> str:
        lookback_block = self._format_tensor_block(lookback, default_message="(not provided)")
        prediction_block = self._format_tensor_block(prediction)
        truth_block = self._format_tensor_block(truth)

        return self.user_prompt_template.format(
            lookback=lookback_block,
            prediction=prediction_block,
            truth=truth_block,
        )

    def _format_tensor_block(
        self,
        tensor: Optional[torch.Tensor],
        *,
        default_message: str = "(empty)",
    ) -> str:
        if tensor is None:
            return default_message
        if tensor.ndim == 0:
            data = [[float(tensor.item())]]
        else:
            data = tensor.view(tensor.shape[0], -1).tolist()
        if not data:
            return default_message

        step = max(1, len(data) // self.max_rows)
        lines = []
        truncated = False
        for idx in range(0, len(data), step):
            sample = ", ".join(f"{value:.{self.precision}f}" for value in data[idx])
            lines.append(f"t={idx}: [{sample}]")
            if len(lines) >= self.max_rows:
                if idx + step < len(data):
                    truncated = True
                break
        if step > 1:
            truncated = True
        if truncated:
            lines.append("...")
        return "\n".join(lines)

    def _query_llm(self, prompt: str) -> float:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [chat_text],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.primary_device) for k, v in inputs.items()}

        pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=pad_token_id,
            )

        output = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        score = self._extract_score(output)
        return score if score is not None else self.fallback_score

    def _extract_score(self, text: str) -> Optional[float]:
        match = self._SCORE_PATTERN.search(text)
        if not match:
            return None
        score = float(match.group(1))
        return max(self.min_score, min(100.0, score))

    def _score_to_loss(self, score: float) -> float:
        return 1.0 / max(self.min_score, score)

    def _infer_primary_device(self) -> torch.device:
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            first_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(first_device, list):
                first_device = first_device[0]
            return torch.device(first_device)
        model_device = getattr(self.model, "device", torch.device("cpu"))
        return torch.device(model_device)
