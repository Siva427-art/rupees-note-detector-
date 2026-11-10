"""Train YOLOv8n on the operator's genuine ₹50 dataset and export ONNX."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from validate_dataset import validate_split

LOGGER = logging.getLogger("train")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="rupee50_yolov8n")
    parser.add_argument("--output", type=Path, default=Path("model.onnx"))
    return parser.parse_args()


def ensure_dataset(data_file: Path) -> None:
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_file}")
    dataset_root = data_file.parent / "dataset"
    for split in ("train", "val", "test"):
        count, errors = validate_split(dataset_root, split, strict=True)
        if errors:
            raise ValueError(f"{split} dataset is invalid: {errors[0]}")
        if count == 0:
            raise ValueError(f"{split} dataset is empty.")


def train(args: argparse.Namespace) -> Path:
    ensure_dataset(args.data)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from error
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(args.project),
        name=args.name,
        single_cls=True,
        exist_ok=False,
    )
    best_weights = args.project / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training completed without best weights: {best_weights}")
    exported = YOLO(str(best_weights)).export(format="onnx", imgsz=args.imgsz,
                                                simplify=True, opset=12)
    exported_path = Path(str(exported))
    if not exported_path.exists():
        raise FileNotFoundError(f"ONNX export did not create a file: {exported_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != args.output.resolve():
        args.output.write_bytes(exported_path.read_bytes())
    LOGGER.info("Training and ONNX export completed: %s", args.output)
    return args.output


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        train(parse_args())
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
