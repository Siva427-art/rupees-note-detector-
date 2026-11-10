"""Capture genuine currency-note images from a webcam.

This script never downloads, synthesizes, or modifies training images. The
operator is responsible for presenting genuine notes and choosing the split.
Press SPACE to save the current frame and Q to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

LOGGER = logging.getLogger("capture_images")
SPLITS = ("train", "val", "test")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--category", default="positive",
                        choices=("positive", "negative"),
                        help="Positive ₹50 notes or negative examples.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def next_filename(directory: Path, category: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    index = 0
    while True:
        candidate = directory / f"{category}_{timestamp}_{index:03d}.jpg"
        if not candidate.exists():
            return candidate
        index += 1


def capture(args: argparse.Namespace) -> int:
    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Unable to open webcam index {args.camera}. Check camera permissions."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    LOGGER.info("Saving %s images to %s", args.category, output_dir.resolve())
    LOGGER.info("SPACE saves a frame; Q quits.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam returned an unreadable frame.")
            raw_frame = frame.copy()
            cv2.putText(
                frame,
                f"{args.split} / {args.category} | SPACE save | Q quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("₹50 dataset capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                filename = next_filename(output_dir, args.category)
                if not cv2.imwrite(str(filename), raw_frame):
                    raise OSError(f"Could not write captured image: {filename}")
                LOGGER.info("Saved %s", filename)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        return capture(args)
    except (OSError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
