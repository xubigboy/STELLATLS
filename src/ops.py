from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np


def aggregate_snapshot_tta(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 4 or p.shape[0] < 1 or p.shape[1] < 1:
        raise ValueError("probabilities must have shape [snapshots, tta, height, width]")
    if np.nanmin(p) < 0 or np.nanmax(p) > 1:
        raise ValueError("probabilities must be in [0, 1]")
    return p.mean(axis=(0, 1)), p.std(axis=(0, 1))


def _filter_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    h, w = mask.shape
    out = np.zeros_like(mask)
    visited = np.zeros_like(mask)
    kept = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            component = []
            while q:
                cy, cx = q.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(component) >= min_area:
                kept += 1
                for cy, cx in component:
                    out[cy, cx] = True
    return out, kept


def apply_tls_cascade(
    probabilities: np.ndarray,
    objectness: float | Sequence[float],
    base_threshold: float = 0.45,
    uncertainty_weight: float = 0.20,
    objectness_gate: float = 0.20,
    min_component_area_px: int = 64,
) -> dict[str, Any]:
    mean_p, uncertainty = aggregate_snapshot_tta(probabilities)
    threshold = base_threshold + uncertainty_weight * uncertainty
    mask = mean_p >= threshold
    obj = float(np.asarray(objectness, dtype=np.float32).mean())
    if obj < objectness_gate:
        mask[:] = False
    mask, component_count = _filter_components(mask, max(1, int(min_component_area_px)))
    confidence = float(mean_p[mask].mean()) if mask.any() else 0.0
    return {
        "mean_probability": mean_p,
        "pixel_uncertainty": uncertainty,
        "local_threshold": threshold,
        "binary_mask": mask,
        "objectness": obj,
        "component_count": component_count,
        "segmentation_confidence": confidence,
    }


def tumor_bed_context(
    semantic_probability: np.ndarray,
    boundary_probability: np.ndarray,
    tls_mask: np.ndarray | None = None,
    semantic_threshold: float = 0.55,
) -> dict[str, Any]:
    semantic = np.asarray(semantic_probability, dtype=np.float32)
    boundary = np.asarray(boundary_probability, dtype=np.float32)
    if semantic.shape != boundary.shape:
        raise ValueError("semantic and boundary probabilities must have equal shape")
    mask = semantic >= semantic_threshold
    result: dict[str, Any] = {
        "semantic_mask": mask,
        "boundary_probability": boundary,
        "foreground_fraction": float(mask.mean()),
        "boundary_mean_probability": float(boundary[mask].mean()) if mask.any() else 0.0,
    }
    if tls_mask is not None:
        tls = np.asarray(tls_mask, dtype=bool)
        if tls.shape != mask.shape:
            raise ValueError("tls_mask must match tumor-bed shape")
        result["tls_tumor_bed_overlap"] = float((tls & mask).sum() / max(int(tls.sum()), 1))
    return result


def ordinal_class_probabilities(logits: Sequence[float]) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (2,):
        raise ValueError("ordinal logits must contain two values")
    q1, q2 = 1.0 / (1.0 + np.exp(-z))
    q1 = float(q1)
    q2 = float(np.clip(q2, 0.0, q1))
    result = np.array([1.0 - q1, q1 - q2, q2], dtype=np.float64)
    return result / result.sum()


def quality_conditioned_route(
    original_probabilities: Sequence[float],
    adapted_probabilities: Sequence[float],
    quality: float,
    quality_threshold: float = 0.90,
    alpha_high: float = 0.00,
    alpha_low: float = 0.75,
) -> np.ndarray:
    p0 = np.asarray(original_probabilities, dtype=np.float32)
    p1 = np.asarray(adapted_probabilities, dtype=np.float32)
    if p0.shape != p1.shape or p0.ndim != 1:
        raise ValueError("expert probabilities must have equal one-dimensional shapes")
    alpha = alpha_high if float(quality) >= quality_threshold else alpha_low
    fused = alpha * p0 + (1.0 - alpha) * p1
    return fused / max(float(fused.sum()), 1e-8)


def entropy(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())

