from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MultiScaleEncoder(nn.Module):
    def __init__(self, width: int = 48, feature_dim: int = 128):
        super().__init__()
        self.stem = ConvBlock(3, width, stride=2)
        self.stage1 = ConvBlock(width, width * 2, stride=2)
        self.stage2 = ConvBlock(width * 2, width * 4, stride=2)
        self.stage3 = ConvBlock(width * 4, width * 6, stride=2)
        self.p2 = nn.Conv2d(width * 4, feature_dim, 1)
        self.p3 = nn.Conv2d(width * 6, feature_dim, 1)
        self.global_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        s = self.stem(x)
        s1 = self.stage1(s)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        p2 = self.p2(s2)
        p3 = self.p3(s3)
        g2 = F.adaptive_avg_pool2d(p2, 1).flatten(1)
        g3 = F.adaptive_avg_pool2d(p3, 1).flatten(1)
        return {"p2": p2, "p3": p3, "global": self.global_proj(torch.cat([g2, g3], dim=1))}


class HierarchicalTokenCompressor(nn.Module):
    def __init__(self, dim: int = 128, keep_ratio: float = 0.25):
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = keep_ratio
        self.score = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1),
        )
        self.token_proj = nn.Linear(dim, dim)

    def forward(self, feature_map: Tensor) -> dict[str, Tensor]:
        b, c, h, w = feature_map.shape
        scores = self.score(feature_map).flatten(1)
        tokens = feature_map.flatten(2).transpose(1, 2)
        k = max(1, min(h * w, int(math.ceil(h * w * self.keep_ratio))))
        top_scores, indices = torch.topk(scores, k=k, dim=1)
        selected = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, c))
        selected = self.token_proj(selected)
        weights = torch.softmax(top_scores, dim=1)
        pooled = (selected * weights.unsqueeze(-1)).sum(dim=1)
        return {"tokens": selected, "pooled": pooled, "scores": top_scores, "indices": indices}


class PathologyAwareMoE(nn.Module):
    def __init__(self, dim: int = 128, experts: int = 3):
        super().__init__()
        if experts < 2:
            raise ValueError("experts must be at least 2")
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, experts),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.softmax(self.router(x), dim=-1)
        expert_values = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (expert_values * weights.unsqueeze(-1)).sum(dim=1)
        return self.norm(x + mixed), weights


class CoordinateGraphTransformer(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, k: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.k = k
        self.coord = nn.Sequential(
            nn.Linear(2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        slide_keys: Sequence[str] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or coords.shape != (x.shape[0], 2):
            raise ValueError("x must be [N,D] and coords must be [N,2]")
        b, d = x.shape
        if b == 1:
            return self.norm(x + self.coord(coords)), torch.zeros((1, 1), device=x.device)
        if slide_keys is None:
            slide_keys = ["batch"] * b
        if len(slide_keys) != b:
            raise ValueError("slide_keys must have one entry per token")
        h = x + self.coord(coords)
        q = F.normalize(self.query(h), dim=-1)
        k = F.normalize(self.key(h), dim=-1)
        v = self.value(h)
        sim = q @ k.transpose(0, 1) / math.sqrt(d)
        same = torch.tensor(
            [[slide_keys[i] == slide_keys[j] for j in range(b)] for i in range(b)],
            device=x.device,
            dtype=torch.bool,
        )
        eye = torch.eye(b, device=x.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -float("inf"))
        same_available = same.sum(dim=1, keepdim=True) > 1
        sim = sim.masked_fill((~same) & same_available, -float("inf"))
        k_eff = min(max(self.k, 1), b - 1)
        values, indices = torch.topk(sim, k=k_eff, dim=1)
        weights = torch.softmax(values.nan_to_num(-1e4), dim=1)
        neighborhood = (v[indices] * weights.unsqueeze(-1)).sum(dim=1)
        h = self.norm(h + self.out(neighborhood))
        refined = self.transformer(h.unsqueeze(0)).squeeze(0)
        return refined, weights


class OrdinalMaturityHead(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.cumulative = nn.Linear(dim, 2)
        self.classifier = nn.Linear(dim, 3)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        cumulative_logits = self.cumulative(x)
        class_logits = self.classifier(x)
        cumulative_prob = torch.sigmoid(cumulative_logits)
        class_prob = torch.softmax(class_logits, dim=1)
        return {
            "cumulative_logits": cumulative_logits,
            "class_logits": class_logits,
            "class_prob": class_prob,
            "ordinal_pred": (cumulative_prob >= 0.5).sum(dim=1),
            "entropy": -(class_prob * (class_prob + 1e-8).log()).sum(dim=1),
            "energy": -torch.logsumexp(class_logits, dim=1),
            "prediction_set": class_prob >= 0.15,
        }


class UnifiedTLSResearchModel(nn.Module):
    """Unified local/context/metadata model for the research pipeline."""

    def __init__(self, meta_dim: int, feature_dim: int = 128, keep_ratio: float = 0.25):
        super().__init__()
        self.local_encoder = MultiScaleEncoder(feature_dim=feature_dim)
        self.context_encoder = MultiScaleEncoder(feature_dim=feature_dim)
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.tls_proposal_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, 1, 1),
        )
        self.tumor_bed_head = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, 1, 1),
        )
        self.compressor = HierarchicalTokenCompressor(feature_dim, keep_ratio=keep_ratio)
        self.moe = PathologyAwareMoE(feature_dim)
        self.fuse = nn.Sequential(
            nn.Linear(feature_dim * 2 + 32, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.graph = CoordinateGraphTransformer(feature_dim)
        self.maturity = OrdinalMaturityHead(feature_dim)

    def forward(
        self,
        local: Tensor,
        context: Tensor,
        meta: Tensor,
        coords: Tensor,
        slide_keys: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        local_features = self.local_encoder(local)
        context_features = self.context_encoder(context)
        tls_logits = F.interpolate(
            self.tls_proposal_head(local_features["p2"]),
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        tumor_input = torch.cat([
            local_features["p2"],
            F.interpolate(
                context_features["p3"],
                size=local_features["p2"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        ], dim=1)
        tumor_logits = F.interpolate(
            self.tumor_bed_head(tumor_input),
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        local_tokens = self.compressor(local_features["p2"])
        context_tokens = self.compressor(context_features["p2"])
        local_moe, local_router = self.moe(local_tokens["pooled"])
        context_moe, context_router = self.moe(context_tokens["pooled"])
        meta_feature = self.meta_encoder(meta)
        fused = self.fuse(torch.cat([
            local_moe + context_moe,
            local_moe - context_moe,
            meta_feature,
        ], dim=1))
        graph_feature, graph_weights = self.graph(fused, coords, slide_keys)
        maturity = self.maturity(graph_feature)
        return {
            "tls_logits": tls_logits,
            "tumor_bed_logits": tumor_logits,
            "tls_probability": torch.sigmoid(tls_logits),
            "tumor_bed_probability": torch.sigmoid(tumor_logits),
            "local_token_scores": local_tokens["scores"],
            "context_token_scores": context_tokens["scores"],
            "local_router": local_router,
            "context_router": context_router,
            "graph_weights": graph_weights,
            "embedding": graph_feature,
            "maturity": maturity,
        }


def ordinal_targets(labels: Tensor) -> Tensor:
    return torch.stack([(labels >= 1).float(), (labels >= 2).float()], dim=1)


def multitask_loss(
    output: dict[str, Any],
    labels: Tensor,
    tls_target: Tensor,
    tumor_target: Tensor,
) -> dict[str, Tensor]:
    maturity = output["maturity"]
    tls_target = tls_target.float()
    tumor_target = tumor_target.float()
    if tls_target.ndim == 3:
        tls_target = tls_target.unsqueeze(1)
    if tumor_target.ndim == 3:
        tumor_target = tumor_target.unsqueeze(1)
    tls_target = F.interpolate(tls_target, size=output["tls_logits"].shape[-2:], mode="nearest")
    tumor_target = F.interpolate(tumor_target, size=output["tumor_bed_logits"].shape[-2:], mode="nearest")
    ordinal = F.binary_cross_entropy_with_logits(
        maturity["cumulative_logits"], ordinal_targets(labels)
    )
    categorical = F.cross_entropy(maturity["class_logits"], labels)
    tls = F.binary_cross_entropy_with_logits(output["tls_logits"], tls_target)
    tumor = F.binary_cross_entropy_with_logits(output["tumor_bed_logits"], tumor_target)
    routers = torch.cat([output["local_router"], output["context_router"]], dim=0)
    mean_router = routers.mean(dim=0)
    balance = ((mean_router - 1.0 / mean_router.numel()) ** 2).mean()
    total = tls + tumor + ordinal + 0.25 * categorical + 0.02 * balance
    return {
        "total": total,
        "tls": tls,
        "tumor_bed": tumor,
        "ordinal": ordinal,
        "categorical": categorical,
        "router_balance": balance,
    }


class DenseEncoder(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, base, 3, 2, 1), nn.BatchNorm2d(base), nn.GELU())
        self.s1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1),
            nn.GELU(),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.GELU(),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),
            nn.GELU(),
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, 2, 1),
            nn.BatchNorm2d(base * 8),
            nn.GELU(),
            nn.Conv2d(base * 8, base * 8, 3, 1, 1),
            nn.GELU(),
        )
        self.out_channels = base * 8

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x0 = self.stem(x)
        x1 = self.s1(x0)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        return x0, x1, x2, x3


class FPNDecoder(nn.Module):
    def __init__(self, base: int = 32, out_channels: int = 64):
        super().__init__()
        self.l3 = nn.Conv2d(base * 8, out_channels, 1)
        self.l2 = nn.Conv2d(base * 4, out_channels, 1)
        self.l1 = nn.Conv2d(base * 2, out_channels, 1)
        self.l0 = nn.Conv2d(base, out_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        x0, x1, x2, x3 = features
        z = self.l3(x3)
        z = F.interpolate(z, size=x2.shape[-2:], mode="bilinear", align_corners=False) + self.l2(x2)
        z = F.interpolate(z, size=x1.shape[-2:], mode="bilinear", align_corners=False) + self.l1(x1)
        z = F.interpolate(z, size=x0.shape[-2:], mode="bilinear", align_corners=False) + self.l0(x0)
        z = self.refine(z)
        return F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)


class TLSInstanceResearchNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.encoder = DenseEncoder(base)
        self.decoder = FPNDecoder(base, 64)
        self.mask_head = nn.Conv2d(64, 1, 1)
        self.aux_mask_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        self.center_head = nn.Conv2d(64, 1, 1)
        self.objectness_head = nn.Linear(base * 8, 1)
        self.box_head = nn.Linear(base * 8, 4)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.encoder(x)
        dense = self.decoder(features)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        return {
            "mask_logits": self.mask_head(dense),
            "aux_mask_logits": self.aux_mask_head(dense),
            "boundary_logits": self.boundary_head(dense),
            "center_logits": self.center_head(dense),
            "objectness_logits": self.objectness_head(pooled),
            "box_raw": self.box_head(pooled),
        }


class TumorBedBoundaryResearchNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.encoder = DenseEncoder(base)
        self.decoder = FPNDecoder(base, 64)
        self.semantic_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        self.context_gate = nn.Sequential(nn.Conv2d(64, 64, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        dense = self.decoder(self.encoder(x))
        gate = self.context_gate(dense)
        return {
            "semantic_logits": self.semantic_head(dense * (1 + gate)),
            "boundary_logits": self.boundary_head(dense),
        }


class MLPExpert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaturityContextOrdinalResearchNet(nn.Module):
    def __init__(self, dim: int = 128, experts: int = 3):
        super().__init__()
        self.local = DenseEncoder(base=24)
        self.context = DenseEncoder(base=24)
        in_dim = 24 * 8
        self.proj = nn.Sequential(nn.Linear(in_dim * 2, dim), nn.LayerNorm(dim), nn.GELU())
        self.experts = nn.ModuleList([MLPExpert(dim) for _ in range(experts)])
        self.router = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, experts))
        self.ordinal = nn.Linear(dim, 2)
        self.classifier = nn.Linear(dim, 3)

    def forward(self, local: Tensor, context: Tensor) -> dict[str, Tensor]:
        l = F.adaptive_avg_pool2d(self.local(local)[-1], 1).flatten(1)
        c = F.adaptive_avg_pool2d(self.context(context)[-1], 1).flatten(1)
        h = self.proj(torch.cat([l, c], dim=1))
        weights = torch.softmax(self.router(h), dim=1)
        values = torch.stack([expert(h) for expert in self.experts], dim=1)
        h = h + (values * weights.unsqueeze(-1)).sum(dim=1)
        ordinal_logits = self.ordinal(h)
        class_logits = self.classifier(h)
        class_prob = torch.softmax(class_logits, dim=1)
        return {
            "embedding": h,
            "router_weights": weights,
            "ordinal_logits": ordinal_logits,
            "class_logits": class_logits,
            "class_prob": class_prob,
            "entropy": -(class_prob * (class_prob + 1e-8).log()).sum(dim=1),
        }


def tversky_loss(logits: Tensor, target: Tensor, alpha: float = 0.7, beta: float = 0.3) -> Tensor:
    probability = torch.sigmoid(logits)
    target = target.float()
    true_positive = (probability * target).sum(dim=(1, 2, 3))
    false_positive = (probability * (1.0 - target)).sum(dim=(1, 2, 3))
    false_negative = ((1.0 - probability) * target).sum(dim=(1, 2, 3))
    score = (true_positive + 1e-6) / (
        true_positive + alpha * false_positive + beta * false_negative + 1e-6
    )
    return (1.0 - score).mean()


def tls_loss(output: dict[str, Tensor], mask: Tensor, objectness: Tensor | None = None, box: Tensor | None = None) -> Tensor:
    target = mask.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = F.interpolate(target, size=output["mask_logits"].shape[-2:], mode="nearest")
    probability = torch.sigmoid(output["mask_logits"])
    inter = (probability * target).sum(dim=(1, 2, 3))
    denom = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    primary_bce = F.binary_cross_entropy_with_logits(output["mask_logits"], target)
    loss = primary_bce + dice + tversky_loss(output["mask_logits"], target)
    if "aux_mask_logits" in output:
        aux_probability = torch.sigmoid(output["aux_mask_logits"])
        aux_inter = (aux_probability * target).sum(dim=(1, 2, 3))
        aux_denom = aux_probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        loss = loss + 0.25 * (1.0 - (2.0 * aux_inter + 1e-6) / (aux_denom + 1e-6)).mean()
    if "boundary_logits" in output:
        dilated = F.max_pool2d(target, 3, 1, 1)
        eroded = -F.max_pool2d(-target, 3, 1, 1)
        boundary = (dilated - eroded).clamp(0, 1)
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(output["boundary_logits"], boundary)
    if "center_logits" in output:
        center_target = F.interpolate(target, size=output["center_logits"].shape[-2:], mode="nearest")
        center_probability = torch.sigmoid(output["center_logits"])
        focal_weight = (1.0 - center_probability).pow(2.0)
        center_focal = (focal_weight * F.binary_cross_entropy_with_logits(
            output["center_logits"], center_target, reduction="none"
        )).mean()
        loss = loss + 0.25 * center_focal
    if objectness is not None:
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
            output["objectness_logits"].squeeze(1), objectness.float()
        )
    if box is not None:
        box_key = "box_raw" if "box_raw" in output else "box_regression"
        loss = loss + 0.1 * F.smooth_l1_loss(output[box_key], box.float())
    return loss


def tumor_bed_loss(output: dict[str, Tensor], mask: Tensor) -> Tensor:
    target = mask.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = F.interpolate(target, size=output["semantic_logits"].shape[-2:], mode="nearest")
    dilated = F.max_pool2d(target, 3, 1, 1)
    eroded = -F.max_pool2d(-target, 3, 1, 1)
    boundary = (dilated - eroded).clamp(0, 1)
    probability = torch.sigmoid(output["semantic_logits"])
    inter = (probability * target).sum(dim=(1, 2, 3))
    denom = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    return (
        F.binary_cross_entropy_with_logits(output["semantic_logits"], target)
        + dice
        + 0.5 * F.binary_cross_entropy_with_logits(output["boundary_logits"], boundary)
    )


def maturity_loss(output: dict[str, Tensor], labels: Tensor) -> Tensor:
    ordinal = torch.stack([(labels >= 1).float(), (labels >= 2).float()], dim=1)
    categorical = F.cross_entropy(output["class_logits"], labels)
    balance = output["class_logits"].new_zeros(())
    if "router_weights" in output:
        router = output["router_weights"].mean(dim=0)
        balance = ((router - 1.0 / router.numel()) ** 2).mean()
    return (
        F.binary_cross_entropy_with_logits(output["ordinal_logits"], ordinal)
        + 0.35 * categorical
        + 0.02 * balance
    )


class RAPIDTLS(TLSInstanceResearchNet):
    """STELLA-TLS public name for the TLS instance segmentation stage."""


class GATETB(TumorBedBoundaryResearchNet):
    """STELLA-TLS public name for the gated tumor-bed segmentation stage."""


class ORDMoE(MaturityContextOrdinalResearchNet):
    """STELLA-TLS public name for the ordinal maturity mixture-of-experts stage."""


class STELLATLS(nn.Module):
    """Three-stage STELLA-TLS model wrapper.

    The wrapper preserves the paper order: RAPID-TLS and GATE-TB run in
    parallel, followed by ORD-MoE on the detected TLS crop and its blurred
    second view. WSI tiling, coordinate registration and patient aggregation
    are provided by the public spatial utilities rather than this module.
    """

    def __init__(self, tls_base: int = 32, tumor_base: int = 32, maturity_dim: int = 128):
        super().__init__()
        self.rapid_tls = RAPIDTLS(base=tls_base)
        self.gate_tb = GATETB(base=tumor_base)
        self.ord_moe = ORDMoE(dim=maturity_dim)

    def forward(
        self,
        tls_patch: Tensor,
        tumor_bed_patch: Tensor,
        maturity_crop: Tensor,
        maturity_blurred_crop: Tensor,
    ) -> dict[str, dict[str, Tensor]]:
        return {
            "rapid_tls": self.rapid_tls(tls_patch),
            "gate_tb": self.gate_tb(tumor_bed_patch),
            "ord_moe": self.ord_moe(maturity_crop, maturity_blurred_crop),
        }


