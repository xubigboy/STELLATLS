from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


def hard_negative_weights(losses: Sequence[float], power: float = 1.5, minimum: float = 0.05) -> np.ndarray:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array")
    if power <= 0:
        raise ValueError("power must be positive")
    shifted = np.maximum(values - np.min(values), 0.0) + minimum
    weights = shifted ** power
    return weights / weights.mean()


def hard_negative_probabilities(losses: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or temperature <= 0:
        raise ValueError("losses must be non-empty and temperature must be positive")
    z = (values - values.max()) / temperature
    p = np.exp(z)
    return p / p.sum()


def tta_flip_probabilities(probabilities: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(item, dtype=np.float32) for item in probabilities]
    if not arrays:
        raise ValueError("probabilities cannot be empty")
    shape = arrays[0].shape
    if any(item.shape != shape for item in arrays):
        raise ValueError("all TTA arrays must have the same shape")
    stack = np.stack(arrays, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def aggregate_maturity_tta(
    ordinal_logits: Sequence[np.ndarray],
    class_probabilities: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    ordinal = np.stack([np.asarray(x, dtype=np.float32) for x in ordinal_logits], axis=0)
    result = {"ordinal_logits": ordinal.mean(axis=0), "ordinal_uncertainty": ordinal.std(axis=0)}
    if class_probabilities is not None:
        probs = np.stack([np.asarray(x, dtype=np.float32) for x in class_probabilities], axis=0)
        result["class_probabilities"] = probs.mean(axis=0)
        result["class_uncertainty"] = probs.std(axis=0)
    return result


def ordinal_prediction(logits: np.ndarray, low: float = 0.5, high: float | None = None) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim < 2 or z.shape[-1] != 2:
        raise ValueError("logits must have final dimension 2")
    if not 0 < low < 1:
        raise ValueError("low must be in (0, 1)")
    high = low if high is None else high
    p = 1.0 / (1.0 + np.exp(-z))
    return ((p[..., 0] >= low).astype(np.int64) + (p[..., 1] >= high).astype(np.int64))


def select_ordinal_thresholds(
    labels: Sequence[int],
    logits: np.ndarray,
    low_grid: Sequence[float] = np.linspace(0.35, 0.65, 13),
    high_grid: Sequence[float] = np.linspace(0.35, 0.75, 17),
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    best: dict[str, Any] | None = None
    for low in low_grid:
        for high in high_grid:
            pred = ordinal_prediction(logits, float(low), float(high))
            accuracy = float(np.mean(pred == y))
            mae = float(np.mean(np.abs(pred - y)))
            score = accuracy - 0.10 * mae
            row = {"low": float(low), "high": float(high), "accuracy": accuracy, "mae": mae, "score": score}
            if best is None or row["score"] > best["score"]:
                best = row
    assert best is not None
    return best


def blend_quality_experts(
    base_probabilities: np.ndarray,
    adapted_probabilities: np.ndarray,
    quality: Sequence[float],
    threshold: float,
    alpha_high: float,
    alpha_low: float,
) -> dict[str, np.ndarray]:
    base = np.asarray(base_probabilities, dtype=np.float32)
    adapted = np.asarray(adapted_probabilities, dtype=np.float32)
    q = np.asarray(quality, dtype=np.float32)
    if base.shape != adapted.shape or base.ndim != 2 or q.shape != (base.shape[0],):
        raise ValueError("expert probabilities must be [N,C] and quality must be [N]")
    alpha = np.where(q >= threshold, alpha_high, alpha_low)[:, None]
    probabilities = alpha * base + (1.0 - alpha) * adapted
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-8, None)
    return {"probabilities": probabilities, "alpha": alpha[:, 0]}


def entropy(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def _pixel_metrics(records: Sequence[dict[str, Any]], threshold_fn: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for record in records:
        probability = np.asarray(record["prob"], dtype=np.float32)
        target = np.asarray(record["target"], dtype=bool)
        pred = probability >= threshold_fn(record)
        counts["tp"] += int((pred & target).sum())
        counts["fp"] += int((pred & ~target).sum())
        counts["fn"] += int((~pred & target).sum())
        counts["tn"] += int((~pred & ~target).sum())
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "n": len(records),
    }


@dataclass
class SlideLevelCalibrator:
    base_threshold: float
    gamma: float
    global_feature: float
    min_threshold: float = 0.25
    max_threshold: float = 0.70

    def threshold(self, slide_feature: float) -> float:
        value = self.base_threshold + self.gamma * (float(slide_feature) - self.global_feature)
        return float(np.clip(value, self.min_threshold, self.max_threshold))

    def to_dict(self) -> dict[str, float]:
        return {
            "base_threshold": self.base_threshold,
            "gamma": self.gamma,
            "global_feature": self.global_feature,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
        }


def fit_slide_calibrator(
    records: Sequence[dict[str, Any]],
    slide_features: dict[str, Sequence[float]],
    base_grid: Sequence[float] = np.arange(0.35, 0.61, 0.025),
    gamma_grid: Sequence[float] = np.arange(-2.0, 2.01, 0.25),
) -> dict[str, Any]:
    slide_medians = {key: float(np.median(values)) for key, values in slide_features.items() if len(values)}
    if not slide_medians:
        raise ValueError("slide_features cannot be empty")
    global_feature = float(np.median(list(slide_medians.values())))
    best: tuple[float, SlideLevelCalibrator, dict[str, Any]] | None = None
    for base in base_grid:
        for gamma in gamma_grid:
            calibrator = SlideLevelCalibrator(float(base), float(gamma), global_feature)
            metrics = _pixel_metrics(
                records,
                lambda record, c=calibrator: c.threshold(
                    slide_medians.get(str(record["slide"]), global_feature)
                ),
            )
            score = float(metrics["dice"])
            if best is None or score > best[0]:
                best = (score, calibrator, metrics)
    assert best is not None
    score, calibrator, metrics = best
    return {
        **calibrator.to_dict(),
        "fit_score": score,
        "fit_metrics": metrics,
        "fit_slides": len(slide_medians),
        "slide_medians": slide_medians,
    }


def calibrated_threshold(calibrator: dict[str, Any], slide_feature: float) -> float:
    return SlideLevelCalibrator(
        base_threshold=float(calibrator["base_threshold"]),
        gamma=float(calibrator["gamma"]),
        global_feature=float(calibrator["global_feature"]),
        min_threshold=float(calibrator.get("min_threshold", 0.25)),
        max_threshold=float(calibrator.get("max_threshold", 0.70)),
    ).threshold(slide_feature)


def evaluate_slide_calibrator(
    records: Sequence[dict[str, Any]],
    slide_features: dict[str, Sequence[float]],
    calibrator: dict[str, Any],
) -> dict[str, Any]:
    medians = {key: float(np.median(values)) for key, values in slide_features.items() if len(values)}
    global_feature = float(calibrator["global_feature"])
    metrics = _pixel_metrics(
        records,
        lambda record: calibrated_threshold(
            calibrator, medians.get(str(record["slide"]), global_feature)
        ),
    )
    return {"metrics": metrics, "slide_medians": medians, "calibrator": calibrator}


