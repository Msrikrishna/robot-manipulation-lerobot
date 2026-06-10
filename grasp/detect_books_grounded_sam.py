"""Most-accurate book boxes: Grounded-SAM (Grounding DINO boxes -> SAM masks).

Why this over plain detection: overlapping / leaning books share too much
axis-aligned box area, so box-NMS either merges distinct books or keeps
duplicates. SAM segments each seed into a pixel-tight mask; deduping by *mask*
IoU separates books that box-IoU cannot, and the final box is read straight off
the mask (tight, and with an oriented minAreaRect for the grasp axis).

Pipeline:
    1. Grounding DINO -> high-recall candidate boxes for "book".
    2. SAM (vit-huge) -> one tight mask per candidate box.
    3. Mask-IoU NMS -> one mask per physical book.
    4. Per book: tight axis-aligned box + oriented box (minAreaRect) + centroid.

Speed: seconds per image (heavy models, CPU/MPS). Low FPS is the trade for
accuracy, as intended.

Usage:
    python grasp/detect_books_grounded_sam.py sample_book_stack_image.png
    python grasp/detect_books_grounded_sam.py img.png --box-thr 0.18 --mask-iou 0.6
    python grasp/detect_books_grounded_sam.py img.png --sam facebook/sam-vit-large
    python grasp/detect_books_grounded_sam.py img.png --out outputs/books_gsam.png --json out.json

First run downloads Grounding DINO (~700MB) and SAM-huge (~2.5GB).
"""
import argparse
import json
import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

GDINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODEL = "facebook/sam-vit-huge"

_PALETTE = [
    (66, 135, 245), (245, 130, 48), (60, 200, 90), (240, 50, 230),
    (255, 225, 25), (70, 240, 240), (230, 25, 75), (170, 110, 40),
    (145, 30, 180), (0, 200, 200),
]


def gdino_boxes(image_pil, prompt, box_thr, text_thr, device):
    """High-recall candidate book boxes from Grounding DINO -> (boxes Nx4, scores N)."""
    proc = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device)
    text = prompt.lower().strip()
    if not text.endswith("."):
        text += "."
    inputs = proc(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


def sam_masks(image_pil, boxes, sam_id, device):
    """One boolean mask per input box via SAM -> array [N, H, W]."""
    if len(boxes) == 0:
        return np.zeros((0,) + image_pil.size[::-1], dtype=bool)
    proc = SamProcessor.from_pretrained(sam_id)
    model = SamModel.from_pretrained(sam_id).to(device)
    inputs = proc(
        image_pil, input_boxes=[boxes.tolist()], return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]  # [N, 1, H, W]
    return masks[:, 0].numpy().astype(bool)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def dedup_masks(masks, scores, mask_iou_thr):
    """Greedy mask-IoU NMS. Returns kept indices (high score first)."""
    order = np.argsort(scores)[::-1]
    keep = []
    for i in order:
        if all(mask_iou(masks[i], masks[j]) <= mask_iou_thr for j in keep):
            keep.append(i)
    return keep


def book_from_mask(mask):
    """Tight box, oriented box, centroid + area from one boolean mask."""
    m = mask.astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pts = cv2.findNonZero(m)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    obb_pts = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(int)
    return {
        "box": [x1, y1, x2, y2],
        "obb": obb_pts.tolist(),                # 4 corners of oriented box
        "centroid": [round(float(cx), 1), round(float(cy), 1)],
        "angle_deg": round(float(angle), 1),    # spine orientation -> gripper yaw
        "area_px": int(m.sum()),
    }


def detect(image_path, prompt="book.", box_thr=0.18, text_thr=0.15,
           mask_iou_thr=0.6, min_area_frac=0.002, sam_id=SAM_MODEL):
    """Grounded-SAM book detection. Returns (detections, BGR image, masks)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w = img_bgr.shape[:2]

    # GDINO has an MPS-unsupported op (cummax); SAM's processor emits float64
    # which MPS rejects. Both run on CUDA if present, else CPU (low FPS is fine).
    gd_dev = sam_dev = "cuda" if torch.cuda.is_available() else "cpu"

    boxes, scores = gdino_boxes(image, prompt, box_thr, text_thr, gd_dev)
    print(f"grounding dino: {len(boxes)} candidate boxes")
    masks = sam_masks(image, boxes, sam_id, sam_dev)

    # drop tiny specks, then dedupe by mask IoU
    areas = masks.reshape(len(masks), -1).sum(1) if len(masks) else np.array([])
    big = [i for i in range(len(masks)) if areas[i] >= min_area_frac * h * w]
    masks, scores = masks[big], scores[big]
    keep = dedup_masks(masks, scores, mask_iou_thr)
    print(f"after mask-IoU dedup: {len(keep)} books")

    dets, kept_masks = [], []
    for rank, i in enumerate(keep):
        info = book_from_mask(masks[i])
        if info is None:
            continue
        info["score"] = round(float(scores[i]), 3)
        dets.append(info)
        kept_masks.append(masks[i])
    return dets, img_bgr, kept_masks


def annotate(img, dets, masks):
    out = img.copy()
    overlay = img.copy()
    for i, (d, m) in enumerate(zip(dets, masks)):
        color = _PALETTE[i % len(_PALETTE)]
        overlay[m] = color
        # oriented box (the grasp-relevant one)
        cv2.polylines(out, [np.array(d["obb"])], True, color, 2, cv2.LINE_AA)
        cx, cy = map(int, d["centroid"])
        cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(out, f"{i} {d['score']:.2f} {d['angle_deg']:.0f}d",
                    (d["box"][0], max(0, d["box"][1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.4, out, 0.6, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--prompt", default="book.")
    ap.add_argument("--box-thr", type=float, default=0.18)
    ap.add_argument("--text-thr", type=float, default=0.15)
    ap.add_argument("--mask-iou", type=float, default=0.6,
                    help="mask-IoU above which two masks are the same book")
    ap.add_argument("--min-area", type=float, default=0.002,
                    help="drop masks smaller than this fraction of the image")
    ap.add_argument("--sam", default=SAM_MODEL)
    ap.add_argument("--out", default="outputs/books_gsam.png")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dets, img, masks = detect(
        args.image, args.prompt, args.box_thr, args.text_thr,
        args.mask_iou, args.min_area, args.sam,
    )
    for i, d in enumerate(dets):
        print(f"  [{i}] score={d['score']:.2f} box={d['box']} "
              f"angle={d['angle_deg']:+.0f}deg centroid={d['centroid']} "
              f"area={d['area_px']}px")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, annotate(img, dets, masks))
        print(f"annotated -> {args.out}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dets, f, indent=2)
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
