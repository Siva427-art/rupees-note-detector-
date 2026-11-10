"""Run real-time ₹50 detection from an exported YOLOv8 ONNX model."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("test")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--temporal-frames", type=int, default=3)
    return parser.parse_args()


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized.shape[1]) // 2
    pad_y = (size - resized.shape[0]) // 2
    canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
    return canvas, scale, pad_x, pad_y


def decode_detections(output: np.ndarray, frame_shape: tuple[int, ...], scale: float,
                      pad_x: int, pad_y: int, confidence: float, nms: float) -> list[tuple[int, int, int, int, float]]:
    predictions = np.squeeze(output)
    if predictions.ndim == 1:
        predictions = predictions[None, :]
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
    if predictions.shape[0] <= 5 and predictions.shape[1] > 5:
        predictions = predictions.T
    boxes: list[list[int]] = []
    scores: list[float] = []
    height, width = frame_shape[:2]
    for row in predictions:
        if row.size < 5:
            continue
        objectness = float(row[4])
        class_score = float(row[5:].max()) if row.size > 5 else 1.0
        score = objectness * class_score
        if score < confidence:
            continue
        center_x, center_y, box_width, box_height = row[:4]
        x1 = int((center_x - box_width / 2 - pad_x) / scale)
        y1 = int((center_y - box_height / 2 - pad_y) / scale)
        x2 = int((center_x + box_width / 2 - pad_x) / scale)
        y2 = int((center_y + box_height / 2 - pad_y) / scale)
        boxes.append([max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)])
        scores.append(score)
    kept = cv2.dnn.NMSBoxes(boxes, scores, confidence, nms)
    indices = np.asarray(kept).reshape(-1) if len(kept) else np.array([], dtype=int)
    return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]) for i in indices]


def run(args: argparse.Namespace) -> int:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train and export a real model first."
        )
    net = cv2.dnn.readNetFromONNX(str(args.model))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera}.")
    recent = deque(maxlen=max(1, args.temporal_frames))
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam returned an unreadable frame.")
            blob_image, scale, pad_x, pad_y = letterbox(frame, args.input_size)
            blob = cv2.dnn.blobFromImage(blob_image, 1 / 255.0, (args.input_size, args.input_size),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            detections = decode_detections(net.forward(), frame.shape, scale, pad_x, pad_y,
                                           args.confidence, args.nms)
            recent.append(bool(detections))
            stable = sum(recent) >= max(1, len(recent) // 2 + 1)
            if detections and stable:
                x1, y1, x2, y2, score = max(detections, key=lambda item: item[4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"₹50 {score:.1%}", (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "₹50 NOT DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("₹50 note detector", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
