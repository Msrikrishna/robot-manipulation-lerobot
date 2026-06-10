"""High-accuracy open-vocab book detection with Grounding DINO (Swin-B).

Grounding DINO is a transformer grounding detector that is markedly more
accurate than YOLO-World at clean, well-separated boxes — at the cost of speed
(seconds per image on CPU/MPS). Low FPS is fine here; we want the best boxes.

Prompt is lowercase, period-separated phrases (Grounding DINO convention), e.g.
"book." or "book. book spine. hardcover book."

Outputs per book: {box: [x1,y1,x2,y2], score, label} + annotated image.

Usage:
    python grasp/detect_books_gdino.py sample_book_stack_image.png
    python grasp/detect_books_gdino.py img.png --prompt "book. book spine."
    python grasp/detect_books_gdino.py img.png --box-thr 0.25 --text-thr 0.20
    python grasp/detect_books_gdino.py img.png --model IDEA-Research/grounding-dino-base
    python grasp/detect_books_gdino.py img.png --out outputs/books_gdino.png --json outputs/books.json

First run downloads the model (~700MB for -base).
"""
import argparse
import json
import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# Grounding DINO uses cummax, unimplemented on MPS -> fall back to CPU for it.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"


def detect(
    image_path: str,
    prompt: str = "book.",
    box_thr: float = 0.25,
    text_thr: float = 0.20,
    nms_iou: float = 0.5,
    model_id: str = DEFAULT_MODEL,
):
    """Run Grounding DINO. Returns (list_of_detections, BGR_image).

    Each detection: {box: [x1,y1,x2,y2], score: float, label: str}.
    Grounding DINO emits one box per matched phrase, so the same book can be
    boxed several times; class-agnostic NMS (nms_iou) collapses duplicates.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Grounding DINO has an MPS-unsupported op (cummax); CUDA or CPU only.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Grounding DINO wants lowercase, period-terminated phrases.
    text = prompt.lower().strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],  # (h, w)
    )[0]

    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    # Class-agnostic NMS to merge the duplicate phrase-groundings per book.
    if len(boxes) > 0:
        keep = nms(boxes.float().cpu(), scores.float().cpu(), nms_iou)
        boxes, scores = boxes[keep], scores[keep]
        labels = [labels[i] for i in keep.tolist()]

    dets = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.tolist()]
        dets.append({
            "box": [x1, y1, x2, y2],
            "score": round(float(score), 3),
            "label": str(label),
        })
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets, img_bgr


def annotate(img, dets):
    out = img.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tag = f"{i}:{d['label']} {d['score']:.2f}"
        cv2.putText(out, tag, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--prompt", default="book.")
    ap.add_argument("--box-thr", type=float, default=0.25,
                    help="min box confidence")
    ap.add_argument("--text-thr", type=float, default=0.20,
                    help="min phrase-grounding confidence")
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="class-agnostic NMS IoU to merge duplicate boxes")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="outputs/books_gdino.png")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dets, img = detect(args.image, args.prompt, args.box_thr, args.text_thr,
                       args.nms_iou, args.model)
    print(f"{len(dets)} detections (prompt={args.prompt!r}, "
          f"box_thr={args.box_thr}, text_thr={args.text_thr}):")
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
