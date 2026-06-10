"""Detect books in an RGB frame with YOLO-World (open-vocabulary).

YOLO-World takes free-text class names, so we can prompt for "book", "book spine",
etc. without any training. "book" is also a COCO class, but the open-vocab prompt
lets us target spines / stacks more precisely for grasping.

Outputs, per detected book:
    - box   : [x1, y1, x2, y2] in pixels (axis-aligned)
    - score : confidence
    - label : matched prompt class
and (optionally) an annotated image saved to --out.

Usage:
    python grasp/detect_books.py sample_book_stack_image.png
    python grasp/detect_books.py img.png --classes "book,book spine,hardcover book"
    python grasp/detect_books.py img.png --conf 0.05 --out outputs/books_det.png
    python grasp/detect_books.py img.png --json outputs/books.json

The first run downloads the YOLO-World weights (yolov8x-worldv2.pt) to the
ultralytics cache.
"""
import argparse
import json
import os

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
# Silence the duplicate-libavdevice objc warning from cv2/av coexisting.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import cv2
import numpy as np
from ultralytics import YOLOWorld

DEFAULT_CLASSES = ["book", "book spine", "stack of books"]


def detect(
    image_path: str,
    classes=DEFAULT_CLASSES,
    conf: float = 0.05,
    iou: float = 0.5,
    imgsz: int = 1280,
    weights: str = "yolov8x-worldv2.pt",
):
    """Run YOLO-World on one image. Returns (list_of_detections, BGR_image).

    Each detection is a dict: {box: [x1,y1,x2,y2], score: float, label: str}.

    imgsz upsamples the input for inference; larger values (e.g. 1280) help
    separate small / tightly-packed book spines at the cost of speed.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    model = YOLOWorld(weights)
    model.set_classes(classes)

    # YOLO-World wants RGB; ultralytics handles BGR->RGB for file paths, but we
    # pass the array so convert explicitly. agnostic_nms dedupes overlapping
    # boxes across the different text-prompt classes.
    results = model.predict(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        conf=conf, iou=iou, imgsz=imgsz, agnostic_nms=True, verbose=False,
    )[0]

    dets = []
    if results.boxes is not None:
        for b in results.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(float).tolist()
            dets.append(
                {
                    "box": [round(v, 1) for v in xyxy],
                    "score": round(float(b.conf[0]), 3),
                    "label": classes[int(b.cls[0])],
                }
            )
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets, img


def annotate(img, dets):
    """Draw boxes + labels on a copy of the BGR image."""
    out = img.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tag = f"{i}:{d['label']} {d['score']:.2f}"
        cv2.putText(
            out, tag, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="comma-separated open-vocab prompts")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--weights", default="yolov8x-worldv2.pt")
    ap.add_argument("--out", default="outputs/books_det.png")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    dets, img = detect(
        args.image, classes, args.conf, args.iou, args.imgsz, args.weights
    )

    print(f"{len(dets)} detections (classes={classes}, conf>={args.conf}):")
    for i, d in enumerate(dets):
        print(f"  [{i}] {d['label']:16s} {d['score']:.3f}  box={d['box']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, annotate(img, dets))
        print(f"annotated -> {args.out}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dets, f, indent=2)
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
