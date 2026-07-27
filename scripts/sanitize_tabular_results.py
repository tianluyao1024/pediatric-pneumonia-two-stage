"""Copy selected CSV outputs while replacing local image paths with image IDs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def sanitize_csv(source: Path, destination: Path) -> None:
    with source.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {source}")
        fields = ["image_id" if name == "path" else name for name in reader.fieldnames]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                if "path" in row:
                    row["image_id"] = Path(row.pop("path")).name
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sanitize_csv(args.source, args.destination)


if __name__ == "__main__":
    main()
