from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.models import MaturityOrdinalModel, TLSInstanceModel, TumorBedModel
from src.research import (
    GATETB,
    ORDMoE,
    RAPIDTLS,
    maturity_loss,
    tls_loss,
    tumor_bed_loss,
)


def tensor_image(x: np.ndarray) -> torch.Tensor:
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[-1] in (1, 3):
        x = np.moveaxis(x, -1, 1)
    if x.ndim == 3:
        x = x[:, None]
    x = x.astype(np.float32)
    if x.max(initial=0) > 1.5:
        x /= 255.0
    return torch.from_numpy(x)


class NpzDataset(Dataset):
    def __init__(self, path: Path, task: str):
        self.data = np.load(path)
        self.task = task
        self.images = tensor_image(self.data["image"]) if task != "maturity" else None
        if task == "maturity":
            self.local = tensor_image(self.data["local"])
            self.context = tensor_image(self.data["context"])
            self.labels = torch.from_numpy(self.data["label"].astype(np.int64))
        elif task == "tls":
            self.masks = torch.from_numpy(self.data["mask"].astype(np.float32))
            self.objectness = torch.from_numpy(self.data["objectness"].astype(np.float32)) if "objectness" in self.data else self.masks.flatten(1).amax(1)
            self.box = torch.from_numpy(self.data["box"].astype(np.float32)) if "box" in self.data else torch.zeros((len(self.images), 4))
        else:
            self.masks = torch.from_numpy(self.data["mask"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels) if self.task == "maturity" else len(self.images)

    def __getitem__(self, index: int):
        if self.task == "maturity":
            return self.local[index], self.context[index], self.labels[index]
        if self.task == "tls":
            return self.images[index], self.masks[index], self.objectness[index], self.box[index]
        return self.images[index], self.masks[index]


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = F.interpolate(target, size=prob.shape[-2:], mode="nearest")
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()


def boundary_target(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    dilated = F.max_pool2d(mask, 3, 1, 1)
    eroded = -F.max_pool2d(-mask, 3, 1, 1)
    return (dilated - eroded).clamp(0, 1)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, task: str) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if task == "tls":
            image, mask, objectness, box = [x.to(device) for x in batch]
            out = model(image)
            loss = tls_loss(out, mask, objectness, box)
        elif task == "tumor_bed":
            image, mask = [x.to(device) for x in batch]
            out = model(image)
            loss = tumor_bed_loss(out, mask)
        else:
            local, context, label = [x.to(device) for x in batch]
            out = model(local, context)
            loss = maturity_loss(out, label)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tls", "tumor_bed", "maturity"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", choices=["research", "compact"], default="research")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = NpzDataset(args.data, args.task)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    compact_models = {"tls": TLSInstanceModel, "tumor_bed": TumorBedModel, "maturity": MaturityOrdinalModel}
    research_models = {
        "tls": RAPIDTLS,
        "tumor_bed": GATETB,
        "maturity": ORDMoE,
    }
    model = (research_models if args.variant == "research" else compact_models)[args.task]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, args.task)
        print(f"epoch={epoch} loss={loss:.6f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": args.task, "variant": args.variant, "model": model.state_dict()}, args.out)


if __name__ == "__main__":
    main()

