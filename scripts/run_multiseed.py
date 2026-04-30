from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_command(command: list[str], gpu_id: str | None) -> None:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    print("running", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def make_seed_config(base_cfg: dict[str, Any], seed: int, output_dir: Path) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["seed"] = seed
    cfg["output_dir"] = str(output_dir)
    return cfg


def summarize_run(run_dir: Path) -> dict[str, Any]:
    history = load_json(run_dir / "history.json")["history"]
    metrics = load_json(run_dir / "test_eval" / "metrics.json")
    best = max(history, key=lambda record: record["score"])
    return {
        "run": run_dir.name,
        "seed": best["config_seed"] if "config_seed" in best else None,
        "epochs": len(history),
        "best_epoch": best["epoch"],
        "best_val_score": best["score"],
        "best_val_cls_acc": best["val"]["cls_acc"],
        "best_val_dice": best["val"]["dice"],
        "test_cls_acc": metrics["classification_accuracy"],
        "test_det_acc": metrics["binary_detection_accuracy"],
        "test_dice": metrics["dice"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple BRISC training seeds from one base config.")
    parser.add_argument("--base-config", default="configs/convnext_tiny_mtl.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", default="outputs/multiseed")
    parser.add_argument("--config-dir", default="outputs/multiseed/configs")
    parser.add_argument("--summary", default="outputs/multiseed/summary.json")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip train/eval when metrics.json already exists.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    base_cfg = load_yaml(base_config_path)
    output_root = Path(args.output_root)
    config_dir = Path(args.config_dir)
    rows = []

    for seed in args.seeds:
        run_name = f"{base_config_path.stem}_seed{seed}"
        run_dir = output_root / run_name
        seed_config_path = config_dir / f"{run_name}.yaml"
        cfg = make_seed_config(base_cfg, seed, run_dir)
        save_yaml(cfg, seed_config_path)

        metrics_path = run_dir / "test_eval" / "metrics.json"
        if not args.skip_existing or not metrics_path.exists():
            run_command(
                [
                    sys.executable,
                    "-m",
                    "brain_tumor_mri.train",
                    "--config",
                    str(seed_config_path),
                    "--device",
                    args.device,
                ],
                args.gpu_id,
            )
            run_command(
                [
                    sys.executable,
                    "-m",
                    "brain_tumor_mri.evaluate",
                    "--checkpoint",
                    str(run_dir / "best.pt"),
                    "--device",
                    args.device,
                ],
                args.gpu_id,
            )
        row = summarize_run(run_dir)
        row["seed"] = seed
        rows.append(row)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": rows}, f, indent=2)
    print(summary_path)


if __name__ == "__main__":
    main()
