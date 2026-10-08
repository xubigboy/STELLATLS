from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DenseEncoder(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.stem = ConvNormAct(3, base, 2)
        self.stage1 = ConvNormAct(base, base * 2, 2)
        self.stage2 = ConvNormAct(base * 2, base * 4, 2)
        self.stage3 = ConvNormAct(base * 4, base * 8, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        f1 = self.stem(x)
        f2 = self.stage1(f1)
        f3 = self.stage2(f2)
        f4 = self.stage3(f3)
        return f1, f2, f3, f4


class FPN(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.l1 = nn.Conv2d(base, 64, 1)
        self.l2 = nn.Conv2d(base * 2, 64, 1)
        self.l3 = nn.Conv2d(base * 4, 64, 1)
        self.l4 = nn.Conv2d(base * 8, 64, 1)
        self.out = ConvNormAct(64, 64)

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f1, f2, f3, f4 = features
        size = f1.shape[-2:]
        x = self.l1(f1)
        x = x + F.interpolate(self.l2(f2), size=size, mode="bilinear", align_corners=False)
        x = x + F.interpolate(self.l3(f3), size=size, mode="bilinear", align_corners=False)
        x = x + F.interpolate(self.l4(f4), size=size, mode="bilinear", align_corners=False)
        return self.out(x)


class TLSInstanceModel(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.encoder = DenseEncoder(base)
        self.fpn = FPN(base)
        self.mask_head = nn.Conv2d(64, 1, 1)
        self.center_head = nn.Conv2d(64, 1, 1)
        self.objectness_head = nn.Linear(base * 8, 1)
        self.box_head = nn.Linear(base * 8, 4)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.encoder(x)
        dense = self.fpn(features)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        return {
            "mask_logits": self.mask_head(dense),
            "center_logits": self.center_head(dense),
            "objectness_logits": self.objectness_head(pooled),
            "box_regression": self.box_head(pooled),
        }


class TumorBedModel(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.encoder = DenseEncoder(base)
        self.fpn = FPN(base)
        self.context_gate = nn.Sequential(nn.Conv2d(64, 64, 1), nn.Sigmoid())
        self.semantic_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        dense = self.fpn(self.encoder(x))
        gated = dense * (1.0 + self.context_gate(dense))
        return {
            "semantic_logits": self.semantic_head(gated),
            "boundary_logits": self.boundary_head(dense),
        }


class MaturityOrdinalModel(nn.Module):
    def __init__(self, base: int = 24, experts: int = 3):
        super().__init__()
        self.local_encoder = DenseEncoder(base)
        self.context_encoder = DenseEncoder(base)
        dim = base * 8 * 2
        self.fuse = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.router = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(experts)])
        self.ordinal_head = nn.Linear(dim, 2)
        self.class_head = nn.Linear(dim, 3)

    def forward(self, local: Tensor, context: Tensor) -> dict[str, Tensor]:
        lf = F.adaptive_avg_pool2d(self.local_encoder(local)[-1], 1).flatten(1)
        cf = F.adaptive_avg_pool2d(self.context_encoder(context)[-1], 1).flatten(1)
        fused = self.fuse(torch.cat([lf, cf], dim=1))
        weights = torch.softmax(self.router(fused), dim=1)
        expert_features = torch.stack([expert(fused) for expert in self.experts], dim=1)
        h = (weights.unsqueeze(-1) * expert_features).sum(dim=1)
        return {
            "router_weights": weights,
            "ordinal_logits": self.ordinal_head(h),
            "class_logits": self.class_head(h),
        }

