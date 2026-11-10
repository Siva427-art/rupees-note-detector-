"""Evaluate the trained detector on the real test split and save reports."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger("evaluate")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("runs/rupee50_yolov8n/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(f"Trained weights not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed. Run pip install -r requirements.txt") from error
    args.reports.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz,
                        plots=True, project=str(args.reports), name="test_metrics",
                        exist_ok=True)
    LOGGER.info("Precision: %.6f", float(metrics.box.mp))
    LOGGER.info("Recall: %.6f", float(metrics.box.mr))
    LOGGER.info("mAP50: %.6f", float(metrics.box.map50))
    LOGGER.info("mAP50-95: %.6f", float(metrics.box.map))
    LOGGER.info("Confusion matrix and plots saved under %s", args.reports)
    LOGGER.info("Metrics are calculated from the supplied test dataset; no scores are fabricated.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        evaluate(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
