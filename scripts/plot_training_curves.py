from __future__ import annotations

import argparse
from pathlib import Path

from analyze_history import plot_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from a run directory with history.json.")
    parser.add_argument("run_dir", help="Run output directory, for example outputs/convnext_tiny_mtl.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to reports/figures/<run_name>_history.png.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.out is None:
        out_path = plot_run(run_dir, Path("reports/figures"))
    else:
        out_path = Path(args.out)
        generated = plot_run(run_dir, out_path.parent)
        if generated != out_path:
            generated.replace(out_path)

    print(out_path)


if __name__ == "__main__":
    main()
