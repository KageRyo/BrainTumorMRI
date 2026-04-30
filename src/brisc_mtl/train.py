from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from brisc_mtl.config import load_config
from brisc_mtl.data import build_samples, class_counts, make_loader, split_train_val
from brisc_mtl.metrics import binary_detection_accuracy
from brisc_mtl.runtime import build_model
from brisc_mtl.utils import device, ensure_dir, save_json, set_seed


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("history", [])


def best_score_from_history(history: list[dict]) -> float:
    if not history:
        return -1.0
    return max(float(record.get("score", -1.0)) for record in history)


def epochs_since_best(history: list[dict], min_delta: float) -> int:
    best = -1.0
    best_epoch_index = -1
    for index, record in enumerate(history):
        score = float(record.get("score", -1.0))
        if score > best + min_delta:
            best = score
            best_epoch_index = index
    if best_epoch_index < 0:
        return 0
    return len(history) - best_epoch_index - 1


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    cls_loss_fn: nn.Module,
    seg_loss_fn: nn.Module,
    dice_metric: DiceMetric,
    cfg: dict,
    dev: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    dice_metric.reset()

    totals = {"loss": 0.0, "cls_loss": 0.0, "seg_loss": 0.0, "cls_acc": 0.0, "det_acc": 0.0}
    count = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "valid")

    for batch in iterator:
        images = batch["image"].to(dev, non_blocking=True)
        masks = batch["mask"].to(dev, non_blocking=True)
        labels = batch["label"].to(dev, non_blocking=True)

        with torch.set_grad_enabled(training), autocast(dev.type, enabled=cfg["train"]["amp"] and dev.type == "cuda"):
            out = model(images)
            cls_loss = cls_loss_fn(out["class_logits"], labels)
            seg_loss = seg_loss_fn(out["mask_logits"], masks)
            loss = cfg["train"]["cls_loss_weight"] * cls_loss + cfg["train"]["seg_loss_weight"] * seg_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(out["mask_logits"])
            dice_metric(y_pred=probs > 0.5, y=masks > 0.5)
            cls_acc = (out["class_logits"].argmax(dim=1) == labels).float().mean().item()
            det_acc = binary_detection_accuracy(out["class_logits"], labels)

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["cls_loss"] += cls_loss.item() * batch_size
        totals["seg_loss"] += seg_loss.item() * batch_size
        totals["cls_acc"] += cls_acc * batch_size
        totals["det_acc"] += det_acc * batch_size
        count += batch_size
        iterator.set_postfix(loss=totals["loss"] / count, cls_acc=totals["cls_acc"] / count)

    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["dice"] = dice_metric.aggregate().item()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BRISC multitask ConvNeXt U-Net.")
    parser.add_argument("--config", default="configs/convnext_base_mtl.yaml")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Training device. Default is 'cuda' so training will not silently fall back to CPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup and exit before the first training epoch.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs from the config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size from the config.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override eval.batch_size from the config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from the config.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from a checkpoint. Use outputs/.../last.pt to continue an interrupted run.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        cfg["eval"]["batch_size"] = args.eval_batch_size
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    set_seed(cfg["seed"])

    out_dir = ensure_dir(cfg["output_dir"])
    dev = device(args.device)

    samples = build_samples(cfg["data_root"], split="train")
    train_samples, val_samples = split_train_val(samples, cfg["data"]["val_fraction"], cfg["seed"])
    save_json(
        {
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "train_class_counts": class_counts(train_samples),
            "val_class_counts": class_counts(val_samples),
        },
        out_dir / "data_summary.json",
    )

    train_loader = make_loader(
        train_samples,
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        training=True,
    )
    val_loader = make_loader(
        val_samples,
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        training=False,
    )

    model = build_model(cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = GradScaler("cuda", enabled=cfg["train"]["amp"] and dev.type == "cuda")
    cls_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    history = load_history(out_dir / "history.json")
    best_score = best_score_from_history(history)
    early_cfg = cfg["train"].get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 0))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    stale_epochs = epochs_since_best(history, early_min_delta) if early_enabled else 0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=dev)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("resume_warning=missing_optimizer_state using_fresh_optimizer=true")
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            print("resume_warning=missing_scheduler_state using_fresh_scheduler=true")
        if "scaler" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint.get("history", history)
        best_score = float(checkpoint.get("best_score", best_score_from_history(history)))
        stale_epochs = int(checkpoint.get("stale_epochs", epochs_since_best(history, early_min_delta)))
        print(f"resume_ok checkpoint={args.resume} start_epoch={start_epoch} best_score={best_score:.4f}")

    if args.dry_run:
        print(
            "dry_run_ok "
            f"device={dev.type} "
            f"train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} "
            f"output_dir={out_dir}"
        )
        return

    if start_epoch > cfg["train"]["epochs"]:
        print(f"training_already_complete start_epoch={start_epoch} configured_epochs={cfg['train']['epochs']}")
        return

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler, cls_loss_fn, seg_loss_fn, dice_metric, cfg, dev
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, scaler, cls_loss_fn, seg_loss_fn, dice_metric, cfg, dev)
        scheduler.step()

        score = 0.5 * val_metrics["cls_acc"] + 0.5 * val_metrics["dice"]
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score}
        history.append(record)
        save_json({"history": history}, out_dir / "history.json")

        print(
            f"epoch={epoch:03d} "
            f"val_cls_acc={val_metrics['cls_acc']:.4f} "
            f"val_det_acc={val_metrics['det_acc']:.4f} "
            f"val_dice={val_metrics['dice']:.4f}"
        )
        improved = score > best_score + early_min_delta
        if improved:
            best_score = score
            stale_epochs = 0
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": val_metrics},
                out_dir / "best.pt",
            )
        elif early_enabled:
            stale_epochs += 1

        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "history": history,
                "best_score": best_score,
                "stale_epochs": stale_epochs,
            },
            out_dir / "last.pt",
        )

        if early_enabled and early_patience > 0 and stale_epochs >= early_patience:
            print(
                f"early_stop epoch={epoch:03d} "
                f"best_score={best_score:.4f} "
                f"stale_epochs={stale_epochs} "
                f"patience={early_patience}"
            )
            break


if __name__ == "__main__":
    main()
