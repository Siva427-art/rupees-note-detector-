# New Indian ₹50 Note Detector

A real-time, single-class computer-vision detector for the new light blue-green Indian ₹50 note. It detects both the Gandhi portrait side and the Hampi stone-chariot side through a webcam, using YOLOv8 and an exported ONNX model.

## Important data policy

This project includes **no images, trained weights, or fabricated metrics**. Capture genuine notes with the supplied webcam tool. Do not download currency images, generate synthetic notes, or claim a model is trained until training has actually completed.

## Structure

```text
dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
capture_images.py
validate_dataset.py
train.py
test.py
evaluate.py
dataset.yaml
requirements.txt
model.onnx                 # created only after real training/export
reports/                   # created by evaluation
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## First run: why `model.onnx` may be missing

`model.onnx` is intentionally not included in this repository. It is generated
only from your own genuine ₹50 note photographs. Therefore this command will
fail until training and export have completed:

```text
ERROR: Model not found: model.onnx. Train and export a real model first.
```

Complete the following sequence before starting the webcam:

1. Capture genuine positive and negative images using `capture_images.py`.
2. Annotate every image and create matching YOLO `.txt` files, including empty
   files for negative images.
3. Run `python validate_dataset.py --strict` and fix every reported error.
4. Run `python train.py --epochs 100 --batch 16 --imgsz 640`.
5. Confirm that `model.onnx` exists in the project root.
6. Run `python test.py --model model.onnx`.

The training command must complete successfully before `model.onnx` exists.
Do not download a currency detector, use synthetic currency images, or create
an empty placeholder model: those would not be a valid trained solution.

## 1. Capture the dataset

Capture approximately 450 images: positive new ₹50 front and back views, plus negative ₹10, ₹20, ₹100, ₹200, ₹500 notes, blue notebooks, green objects, and empty backgrounds. Include different distances, angles, lighting, clutter, and backgrounds. Use genuine objects only.

```bash
python capture_images.py --split train --category positive
python capture_images.py --split val --category positive
python capture_images.py --split test --category positive
python capture_images.py --split train --category negative
python capture_images.py --split val --category negative
python capture_images.py --split test --category negative
```

Press **SPACE** to save and **Q** to quit. The captured frame is saved with a unique timestamped name. Keep a balanced split; a practical starting point is 70% train, 20% validation, 10% test.

The capture tool saves images directly in the split root with a category-prefixed filename. Do not add a label file for a negative image when using `--strict` only if your training pipeline accepts empty-label files. For reliable YOLO training, create an empty `.txt` file with the same basename for every negative image.

## 2. Annotate

Use LabelImg with YOLO format. Draw boxes only around genuine ₹50 notes. Use exactly one class:

```text
0 new_50_rupee_note
```

Each label line is `class_id center_x center_y width height`, normalized to 0–1. Negative images have no boxes and should have an empty matching `.txt` file. Validate before training:

```bash
python validate_dataset.py --strict
```

## 3. Train and export ONNX

The script checks every split, trains YOLOv8n, saves Ultralytics results under `runs/`, and exports the actual best weights to `model.onnx`.

```bash
python train.py --epochs 100 --batch 16 --imgsz 640
```

Training requires `yolov8n.pt`; Ultralytics may download this model checkpoint as a framework dependency. This project does not include or claim a trained detector until this command succeeds.

## 4. Webcam detection

The company-required command works after `model.onnx` has been created:

```bash
python test.py --model model.onnx
```

The application uses OpenCV DNN, confidence thresholding, class-aware non-maximum suppression, FPS display, and a short temporal confidence filter. Temporal filtering reduces flicker; it cannot guarantee correct classification. Press **Q** to exit.

## 5. Evaluation

Run evaluation against the held-out test set:

```bash
python evaluate.py --model runs/rupee50_yolov8n/weights/best.pt
```

Ultralytics calculates precision, recall, mAP50, and mAP50-95. Confusion matrices, plots, and false-positive/negative artifacts are written under `reports/test_metrics/`. Performance on other denominations is measured only when those genuine negative examples exist in the test split. Never invent or manually edit scores.

## Architecture

Webcam → OpenCV frame capture → letterbox preprocessing → YOLOv8 ONNX/OpenCV DNN → confidence filtering → NMS → temporal stability filter → annotated display.

## Screenshots

Add screenshots of the capture UI, annotation examples, training plots, evaluation confusion matrix, and successful front/back webcam detections after collecting real project evidence.

## Limitations

- Accuracy depends on genuine, varied, correctly annotated data.
- Severe blur, occlusion, glare, unusual backgrounds, and poor lighting can cause misses or false positives.
- A single-class detector cannot identify which side of the note is visible.
- Temporal filtering improves display stability but is not a security guarantee.
- Webcam performance depends on camera resolution, CPU/GPU, and ONNX runtime.

## Future enhancements

Add side-specific classes, calibration and camera-quality checks, hard-negative mining, model quantization benchmarking, automated dataset statistics, multi-camera support, and a controlled deployment test protocol.
