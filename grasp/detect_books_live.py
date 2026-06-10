"""Real-time YOLO-World book detection on a live camera feed.

Opens an OpenCV camera, runs open-vocab detection every frame, and draws boxes
in a live window with an FPS readout. Same prompts/knobs as detect_books.py but
defaults to a lighter model + smaller imgsz so it keeps up with video.

Usage:
    python grasp/detect_books_live.py                       # camera 0, defaults
    python grasp/detect_books_live.py --camera 1
    python grasp/detect_books_live.py --classes "book,book spine"
    python grasp/detect_books_live.py --conf 0.10 --imgsz 640
    python grasp/detect_books_live.py --weights yolov8x-worldv2.pt   # heavier/slower
    python grasp/detect_books_live.py --record outputs/live.mp4      # save video

Keys:  q / Esc = quit
The first run downloads the YOLO-World weights to the ultralytics cache.
"""
import argparse
import os
import time

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import cv2
import torch
from ultralytics import YOLOWorld

from detect_books import DEFAULT_CLASSES, annotate


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="comma-separated open-vocab prompts")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size; 640 is fast, 1280 separates packed books")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt",
                    help="yolov8s/m/l/x-worldv2.pt (s=fastest, x=most accurate)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--record", default=None, help="optional path to save annotated mp4")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    device = pick_device()
    print(f"device: {device} | weights: {args.weights} | classes: {classes}")

    model = YOLOWorld(args.weights)
    model.set_classes(classes)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {args.camera}")

    writer = None
    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 20.0, (args.width, args.height))

    fps = 0.0
    win = "books (YOLO-World) - q to quit"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break

            t0 = time.time()
            results = model.predict(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                agnostic_nms=True, device=device, verbose=False,
            )[0]

            dets = []
            if results.boxes is not None:
                for b in results.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().astype(float).tolist()
                    dets.append({
                        "box": [round(v, 1) for v in xyxy],
                        "score": round(float(b.conf[0]), 3),
                        "label": classes[int(b.cls[0])],
                    })
            dets.sort(key=lambda d: d["score"], reverse=True)

            # exponential-moving-average FPS for a stable readout
            inst = 1.0 / max(time.time() - t0, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            out = annotate(frame, dets)
            cv2.putText(out, f"{fps:4.1f} FPS | {len(dets)} books",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(out)
            cv2.imshow(win, out)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or Esc
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"saved -> {args.record}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
