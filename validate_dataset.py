"""Validate image/YOLO-label consistency before training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

LOGGER = logging.getLogger("validate_dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--strict", action="store_true",
                        help="Treat missing labels as errors, including negatives.")
    return parser.parse_args()


def validate_label(label_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [f"{label_path}: cannot read label ({error})"]
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{label_path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id, *coordinates = [float(value) for value in fields]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric label")
            continue
        if class_id != 0 or any(value < 0 or value > 1 for value in coordinates):
            errors.append(f"{label_path}:{line_number}: invalid class or normalized box")
        if coordinates[2] == 0 or coordinates[3] == 0:
            errors.append(f"{label_path}:{line_number}: box has zero area")
    return errors


def validate_split(dataset_root: Path, split: str, strict: bool) -> tuple[int, list[str]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    errors: list[str] = []
    if not image_dir.exists():
        return 0, [f"Missing image directory: {image_dir}"]
    if not label_dir.exists():
        return 0, [f"Missing label directory: {label_dir}"]
    images = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    image_stems = {path.stem for path in images}
    label_paths = list(label_dir.rglob("*.txt"))
    label_stems = {path.stem for path in label_paths}
    for image in images:
        if strict and image.stem not in label_stems:
            errors.append(f"Missing annotation for image: {image}")
        if cv2.imread(str(image)) is None:
            errors.append(f"Unreadable image: {image}")
    for label in label_paths:
        if label.stem not in image_stems:
            errors.append(f"Label has no matching image: {label}")
        errors.extend(validate_label(label))
    return len(images), errors


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    total = 0
    errors: list[str] = []
    for split in ("train", "val", "test"):
        count, split_errors = validate_split(args.dataset_root, split, args.strict)
        total += count
        LOGGER.info("%s: %d images", split, count)
        errors.extend(split_errors)
    if errors:
        for error in errors:
            LOGGER.error("%s", error)
        return 1
    if total == 0:
        LOGGER.error("Dataset contains no images. Capture genuine images before training.")
        return 1
    LOGGER.info("Dataset validation passed: %d images", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
