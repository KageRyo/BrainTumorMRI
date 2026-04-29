from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_run(run_dir: Path) -> dict[str, Any]:
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "test_eval" / "metrics.json"
    history = load_json(history_path).get("history", []) if history_path.exists() else []
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    best_val = max(history, key=lambda record: record.get("score", -1.0), default={})
    last_val = history[-1] if history else {}
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "epochs": len(history),
        "best_epoch": best_val.get("epoch"),
        "best_val_score": best_val.get("score"),
        "best_val_cls_acc": best_val.get("val", {}).get("cls_acc"),
        "best_val_det_acc": best_val.get("val", {}).get("det_acc"),
        "best_val_dice": best_val.get("val", {}).get("dice"),
        "last_epoch": last_val.get("epoch"),
        "test_cls_acc": metrics.get("classification_accuracy"),
        "test_det_acc": metrics.get("binary_detection_accuracy"),
        "test_dice": metrics.get("dice"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def make_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "epochs",
        "best epoch",
        "best val score",
        "val cls acc",
        "val det acc",
        "val dice",
        "test cls acc",
        "test det acc",
        "test dice",
    ]
    keys = [
        "name",
        "epochs",
        "best_epoch",
        "best_val_score",
        "best_val_cls_acc",
        "best_val_det_acc",
        "best_val_dice",
        "test_cls_acc",
        "test_det_acc",
        "test_dice",
    ]
    lines = [
        "# BRISC Model Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in keys) + " |")
    lines.append("")
    lines.append("Validation score = 0.5 * classification accuracy + 0.5 * Dice.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BRISC training and test metrics across output directories.")
    parser.add_argument("runs", nargs="+", help="Output directories such as outputs/convnext_base_mtl.")
    parser.add_argument("--out", default="outputs/model_comparison.md", help="Markdown report path.")
    args = parser.parse_args()

    rows = [read_run(Path(run)) for run in args.runs]
    report = make_markdown(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
