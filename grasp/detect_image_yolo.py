"""Run prompt-based YOLO-World detection on a single image file.

Same inference path as the web UI (grasp/web_detect_yolo.py): same small/large
model names and the same defaults, so CLI and UI results match.

Usage:
    python grasp/detect_image_yolo.py path/to/img.png
    python grasp/detect_image_yolo.py img.png --prompt "book, book spine"
    python grasp/detect_image_yolo.py img.png --model large --imgsz 1280 --conf 0.05
    python grasp/detect_image_yolo.py img.png --out outputs/det.png --json outputs/det.json

Models: small=yolov8s-worldv2.pt (fast), large=yolov8x-worldv2.pt (accurate).
First run per model downloads the weights to the ultralytics cache.
"""
import argparse
import json
import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import cv2

from web_detect_yolo import run_detection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--prompt", default="book, book spine, stack of books",
                    help="comma-separated open-vocab class prompts")
    ap.add_argument("--model", choices=["small", "large"], default="small")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="outputs/detect_image.png",
                    help="annotated image output (set '' to skip)")
    ap.add_argument("--json", default=None, help="optional path to dump detections")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    classes = [c.strip() for c in args.prompt.split(",") if c.strip()] or ["book"]
    dets, annotated = run_detection(
        img, classes=classes, model_key=args.model,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz,
    )

    print(f"{len(dets)} detection(s)  model={args.model}  classes={classes}  conf>={args.conf}")
    for i, d in enumerate(dets):
        print(f"  [{i}] {d['label']:18s} {d['score']:.3f}  box={d['box']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, annotated)
        print(f"annotated -> {args.out}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dets, f, indent=2)
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
