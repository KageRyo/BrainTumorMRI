from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BRISC 2025 from Kaggle.")
    parser.add_argument("--out", default="data/brisc2025", help="Destination directory.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into --out. By default prints the Kaggle cache path.",
    )
    args = parser.parse_args()

    path = Path(kagglehub.dataset_download("briscdataset/brisc2025"))
    print(f"Path to dataset files: {path}")

    if args.copy:
        out = Path(args.out)
        if out.exists():
            raise FileExistsError(f"{out} already exists. Remove it or choose another --out.")
        shutil.copytree(path, out)
        print(f"Copied dataset to: {out}")


if __name__ == "__main__":
    main()
