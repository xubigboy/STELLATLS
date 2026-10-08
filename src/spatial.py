from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def generate_candidate_tiles(
    tissue_mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_fraction: float = 0.50,
    wsi_scale: float = 8.0,
) -> list[dict[str, Any]]:
    """Generate coordinate-preserving tissue-qualified candidate tiles."""
    mask = np.asarray(tissue_mask, dtype=bool)
    if mask.ndim != 2 or patch_size <= 0 or stride <= 0:
        raise ValueError("tissue_mask must be 2D and patch_size/stride must be positive")
    if not 0 <= min_tissue_fraction <= 1:
        raise ValueError("min_tissue_fraction must be in [0, 1]")
    height, width = mask.shape
    ys = list(range(0, max(height - patch_size, 0) + 1, stride))
    xs = list(range(0, max(width - patch_size, 0) + 1, stride))
    ys.append(max(height - patch_size, 0))
    xs.append(max(width - patch_size, 0))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    tiles: list[dict[str, Any]] = []
    for y in ys:
        for x in xs:
            y1 = min(y + patch_size, height)
            x1 = min(x + patch_size, width)
            local = mask[y:y1, x:x1]
            coverage = float(local.mean()) if local.size else 0.0
            if coverage >= min_tissue_fraction:
                tiles.append({
                    "x0": int(x),
                    "y0": int(y),
                    "width": int(x1 - x),
                    "height": int(y1 - y),
                    "coverage": coverage,
                    "wsi_scale": float(wsi_scale),
                })
    return tiles


def map_patch_point_to_wsi(x: float, y: float, x0: float, y0: float, scale: float) -> tuple[float, float]:
    """Map a local patch point to the WSI coordinate frame."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return float(x0 + scale * x), float(y0 + scale * y)


def assign_spatial_compartment(signed_distance_um: float, boundary_band_um: float = 500.0) -> str:
    """Assign peritumoral, invasive-margin or intratumoral compartment."""
    distance = float(signed_distance_um)
    band = abs(float(boundary_band_um))
    if distance < -band:
        return "peritumoral"
    if distance > band:
        return "intratumoral"
    return "invasive_margin"


def build_descriptive_tls_graph(nodes: Sequence[dict[str, Any]], k: int = 4) -> list[dict[str, Any]]:
    """Build a non-trainable coordinate-aware kNN graph for TLS descriptors."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not nodes:
        return []
    points = []
    for node in nodes:
        if "x_wsi" in node and "y_wsi" in node:
            points.append([float(node["x_wsi"]), float(node["y_wsi"])])
        elif "x" in node and "y" in node:
            points.append([float(node["x"]), float(node["y"])])
        else:
            raise ValueError("each node needs x_wsi/y_wsi or x/y")
    coords = np.asarray(points, dtype=np.float64)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    edges: list[dict[str, Any]] = []
    for source in range(len(nodes)):
        order = np.argsort(distances[source])
        for target in order[1 : min(k + 1, len(nodes))]:
            edges.append({"source": int(source), "target": int(target), "distance": float(distances[source, target])})
    return edges


def patient_level_endpoints(
    nodes: Sequence[dict[str, Any]],
    evaluable_tissue_area: float,
    tumor_bed_area: float,
) -> dict[str, float]:
    """Calculate the STELLA-TLS patient-level spatial-maturity endpoints."""
    count = len(nodes)
    area = max(float(evaluable_tissue_area), 1e-12)
    tumor_area = max(float(tumor_bed_area), 1e-12)
    compartment_order = ("peritumoral", "invasive_margin", "intratumoral")
    spatial_weight = {name: index + 1 for index, name in enumerate(compartment_order)}
    compartment_counts = {name: 0 for name in compartment_order}
    sm_score = 0.0
    mature_count = 0
    for node in nodes:
        compartment = str(node.get("compartment", "invasive_margin"))
        if compartment not in compartment_counts:
            compartment = "invasive_margin"
        compartment_counts[compartment] += 1
        probabilities = np.asarray(node.get("maturity_probabilities", []), dtype=np.float64)
        if probabilities.shape == (3,):
            probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
            maturity_weight = float(np.dot(probabilities, np.array([1.0, 2.0, 3.0])))
            predicted = int(np.argmax(probabilities))
        else:
            predicted = int(node.get("maturity_index", 0))
            maturity_weight = float(np.clip(predicted + 1, 1, 3))
        mature_count += int(predicted == 2)
        sm_score += float(node.get("area", 0.0)) * maturity_weight * spatial_weight[compartment]
    proportions = np.asarray([compartment_counts[name] for name in compartment_order], dtype=np.float64)
    proportions /= max(float(proportions.sum()), 1.0)
    spatial_entropy = float(-(proportions * np.log(np.clip(proportions, 1e-12, 1.0))).sum() / np.log(3.0))
    return {
        "tls_count": float(count),
        "tls_density": float(count / area),
        "mature_tls_ratio": float(mature_count / max(count, 1)),
        "spatial_entropy": spatial_entropy,
        "sm_tls_score": float(sm_score / tumor_area),
        "peritumoral_count": float(compartment_counts["peritumoral"]),
        "invasive_margin_count": float(compartment_counts["invasive_margin"]),
        "intratumoral_count": float(compartment_counts["intratumoral"]),
    }

