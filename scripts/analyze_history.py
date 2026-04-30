from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "history.json"
    with history_path.open("r", encoding="utf-8") as f:
        return json.load(f)["history"]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    history = load_history(run_dir)
    best = max(history, key=lambda record: record["score"])
    final = history[-1]
    return {
        "run": run_dir.name,
        "epochs": len(history),
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "best_val_cls_acc": best["val"]["cls_acc"],
        "best_val_dice": best["val"]["dice"],
        "best_cls_gap": best["train"]["cls_acc"] - best["val"]["cls_acc"],
        "best_dice_gap": best["train"]["dice"] - best["val"]["dice"],
        "final_score": final["score"],
        "final_val_dice": final["val"]["dice"],
        "final_dice_gap": final["train"]["dice"] - final["val"]["dice"],
    }


def plot_run(run_dir: Path, out_dir: Path) -> Path:
    history = load_history(run_dir)
    epochs = [record["epoch"] for record in history]
    best = max(history, key=lambda record: record["score"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    series = [
        ("loss", "Loss"),
        ("cls_acc", "Classification Accuracy"),
        ("dice", "Dice"),
    ]
    for axis, (metric, title) in zip(axes, series, strict=True):
        axis.plot(epochs, [record["train"][metric] for record in history], label=f"train {metric}")
        axis.plot(epochs, [record["val"][metric] for record in history], label=f"val {metric}")
        axis.axvline(best["epoch"], color="gray", linestyle="--", linewidth=1, label="best epoch")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle(run_dir.name)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_dir.name}_history.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def make_markdown(rows: list[dict[str, Any]], figures: list[Path]) -> str:
    headers = [
        "run",
        "epochs",
        "best epoch",
        "best score",
        "best val cls",
        "best val dice",
        "best cls gap",
        "best dice gap",
        "final score",
        "final dice gap",
    ]
    keys = [
        "run",
        "epochs",
        "best_epoch",
        "best_score",
        "best_val_cls_acc",
        "best_val_dice",
        "best_cls_gap",
        "best_dice_gap",
        "final_score",
        "final_dice_gap",
    ]
    lines = [
        "# Training History Analysis",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in keys) + " |")

    lines.extend(
        [
            "",
            "Gap values are train metric minus validation metric. "
            "Larger positive Dice gaps indicate more segmentation overfitting risk.",
            "",
            "## Curves",
            "",
        ]
    )
    for figure in figures:
        lines.append(f"![{figure.stem}]({figure.as_posix()})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze training histories and plot train/validation curves.")
    parser.add_argument("runs", nargs="+", help="Output directories with history.json files.")
    parser.add_argument("--out", default="reports/history_analysis.md", help="Markdown report path.")
    parser.add_argument("--figures-dir", default="reports/figures", help="Directory for curve figures.")
    args = parser.parse_args()

    out_path = Path(args.out)
    figures_dir = Path(args.figures_dir)
    rows = [summarize_run(Path(run)) for run in args.runs]
    figures = [plot_run(Path(run), figures_dir) for run in args.runs]
    relative_figures = [figure.relative_to(out_path.parent) for figure in figures]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(make_markdown(rows, relative_figures), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
