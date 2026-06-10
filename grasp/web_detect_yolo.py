"""Tiny web UI for prompt-based YOLO-World book/object detection.

Snap a frame from your webcam, type open-vocab prompts, pick the small or large
model, and get bounding boxes back on the right — no retraining, just text.

Runs on Python's stdlib http.server (no Flask/FastAPI needed). Models are loaded
lazily and cached, so the first detection per model is slow, the rest are fast.

Usage:
    python grasp/web_detect_yolo.py            # serves http://localhost:8000
    python grasp/web_detect_yolo.py --port 8080

Then open the URL in a browser, allow camera access, snap + detect.
"""
import argparse
import base64
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import cv2
import numpy as np
from ultralytics import YOLOWorld

from detect_books import annotate

# label -> weights file. Both already live in the repo root.
MODELS = {
    "small": "yolov8s-worldv2.pt",
    "large": "yolov8x-worldv2.pt",
}
_MODEL_CACHE = {}


def get_model(key: str) -> YOLOWorld:
    """Lazy-load + cache a YOLO-World model by 'small'/'large' key."""
    weights = MODELS.get(key, MODELS["small"])
    if weights not in _MODEL_CACHE:
        print(f"loading {weights} ...")
        _MODEL_CACHE[weights] = YOLOWorld(weights)
    return _MODEL_CACHE[weights]


def run_detection(img_bgr, classes, model_key, conf, iou, imgsz):
    """Run YOLO-World on a BGR frame. Returns (dets, annotated_bgr)."""
    model = get_model(model_key)
    model.set_classes(classes)
    results = model.predict(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        conf=conf, iou=iou, imgsz=imgsz, agnostic_nms=True, verbose=False,
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
    return dets, annotate(img_bgr, dets)


def _decode_data_url(data_url: str):
    """data:image/png;base64,.... -> BGR numpy image."""
    b64 = re.sub(r"^data:image/\w+;base64,", "", data_url)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpg(img_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/detect":
            self._send(404, "not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            img = _decode_data_url(req["image"])
            if img is None:
                raise ValueError("could not decode image")
            classes = [c.strip() for c in req.get("prompt", "book").split(",") if c.strip()]
            if not classes:
                classes = ["book"]
            dets, annotated = run_detection(
                img,
                classes=classes,
                model_key=req.get("model", "small"),
                conf=float(req.get("conf", 0.05)),
                iou=float(req.get("iou", 0.5)),
                imgsz=int(req.get("imgsz", 1280)),
            )
            self._send(200, json.dumps({
                "dets": dets,
                "image": _encode_jpg(annotated),
                "classes": classes,
            }))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO-World Detector</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e9ef; --mut:#8b93a7; --acc:#4f9dff; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:16px 22px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:17px; font-weight:600; }
  header p { margin:3px 0 0; color:var(--mut); font-size:12px; }
  .wrap { display:grid; grid-template-columns:1fr 1fr; gap:18px; padding:18px; align-items:start; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
  .panel h2 { margin:0 0 12px; font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); }
  video, #snap, #result { width:100%; border-radius:8px; background:#000; display:block; }
  .controls { display:flex; flex-direction:column; gap:12px; margin-top:14px; }
  label { font-size:12px; color:var(--mut); display:block; margin-bottom:4px; }
  input[type=text], select { width:100%; padding:9px 11px; background:#0e1014; color:var(--fg);
    border:1px solid var(--line); border-radius:7px; font-size:13px; }
  .row { display:flex; gap:10px; }
  .row > div { flex:1; }
  .btns { display:flex; gap:10px; }
  button { flex:1; padding:11px; border:0; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; }
  #snapBtn { background:#2a2f3a; color:var(--fg); }
  #detectBtn { background:var(--acc); color:#fff; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .hint { font-size:11px; color:var(--mut); }
  table { width:100%; border-collapse:collapse; margin-top:12px; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--mut); font-weight:500; font-size:11px; text-transform:uppercase; }
  .box-coords { color:var(--mut); font-family:ui-monospace,monospace; font-size:11px; }
  .status { font-size:12px; color:var(--mut); min-height:16px; }
  .empty { color:var(--mut); padding:20px 0; text-align:center; }
</style>
</head>
<body>
<header>
  <h1>YOLO-World prompt detector</h1>
  <p>Snap a webcam frame, type open-vocab prompts (comma-separated), pick a model, get boxes.</p>
</header>

<div class="wrap">
  <!-- LEFT: capture + controls -->
  <div class="panel">
    <h2>Capture</h2>
    <video id="cam" autoplay playsinline muted></video>
    <img id="shot" alt="captured frame" style="display:none">
    <canvas id="snap" style="display:none"></canvas>

    <div class="controls">
      <div>
        <label>Prompts (comma-separated open-vocab classes)</label>
        <input id="prompt" type="text" value="book, book spine, stack of books">
      </div>
      <div class="row">
        <div>
          <label>Model</label>
          <select id="model">
            <option value="small">Small (yolov8s — fast)</option>
            <option value="large">Large (yolov8x — accurate)</option>
          </select>
        </div>
        <div>
          <label>Image size</label>
          <select id="imgsz">
            <option value="640">640 (fast)</option>
            <option value="1280" selected>1280 (packed objects)</option>
          </select>
        </div>
        <div>
          <label>Min conf</label>
          <input id="conf" type="text" value="0.05">
        </div>
      </div>
      <div class="btns">
        <button id="snapBtn">Snap frame</button>
        <button id="retakeBtn" disabled>Retake</button>
        <button id="detectBtn" disabled>Detect</button>
      </div>
      <div class="status" id="status">Allow camera access to begin.</div>
    </div>
  </div>

  <!-- RIGHT: results -->
  <div class="panel">
    <h2>Results</h2>
    <img id="result" alt="detection result">
    <div id="resultArea">
      <div class="empty">No detection yet — snap a frame and hit Detect.</div>
    </div>
  </div>
</div>

<script>
const cam = document.getElementById('cam');
const shot = document.getElementById('shot');
const snap = document.getElementById('snap');
const resultImg = document.getElementById('result');
const resultArea = document.getElementById('resultArea');
const statusEl = document.getElementById('status');
const snapBtn = document.getElementById('snapBtn');
const retakeBtn = document.getElementById('retakeBtn');
const detectBtn = document.getElementById('detectBtn');
let snapped = null;  // data URL of the captured frame — held until Retake

resultImg.style.display = 'none';

navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
  .then(s => { cam.srcObject = s; statusEl.textContent = 'Camera ready. Snap a frame.'; })
  .catch(e => { statusEl.textContent = 'Camera error: ' + e.message; });

snapBtn.onclick = () => {
  const w = cam.videoWidth, h = cam.videoHeight;
  if (!w) { statusEl.textContent = 'Camera not ready yet.'; return; }
  snap.width = w; snap.height = h;
  snap.getContext('2d').drawImage(cam, 0, 0, w, h);
  snapped = snap.toDataURL('image/jpeg', 0.92);
  // Freeze the still in place of the live feed so it's visibly "held".
  shot.src = snapped;
  shot.style.display = 'block';
  cam.style.display = 'none';
  snapBtn.disabled = true;
  retakeBtn.disabled = false;
  detectBtn.disabled = false;
  statusEl.textContent = `Frame held (${w}x${h}). Change the prompt and Detect — same frame is reused.`;
};

retakeBtn.onclick = () => {
  // Drop the held frame and go back to live to capture a new one.
  snapped = null;
  shot.style.display = 'none';
  cam.style.display = 'block';
  snapBtn.disabled = false;
  retakeBtn.disabled = true;
  detectBtn.disabled = true;
  statusEl.textContent = 'Live. Snap a new frame.';
};

detectBtn.onclick = async () => {
  if (!snapped) return;
  detectBtn.disabled = true;
  statusEl.textContent = 'Running detection...';
  try {
    const res = await fetch('/detect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: snapped,
        prompt: document.getElementById('prompt').value,
        model: document.getElementById('model').value,
        imgsz: document.getElementById('imgsz').value,
        conf: document.getElementById('conf').value,
      }),
    });
    const data = await res.json();
    if (data.error) { statusEl.textContent = 'Error: ' + data.error; detectBtn.disabled = false; return; }
    resultImg.src = data.image;
    resultImg.style.display = 'block';
    renderTable(data.dets);
    statusEl.textContent = `${data.dets.length} detection(s) for [${data.classes.join(', ')}].`;
  } catch (e) {
    statusEl.textContent = 'Request failed: ' + e.message;
  }
  detectBtn.disabled = false;
};

function renderTable(dets) {
  if (!dets.length) { resultArea.innerHTML = '<div class="empty">No objects matched the prompt.</div>'; return; }
  let rows = dets.map((d, i) =>
    `<tr><td>${i}</td><td>${d.label}</td><td>${d.score.toFixed(3)}</td>
     <td class="box-coords">[${d.box.join(', ')}]</td></tr>`).join('');
  resultArea.innerHTML =
    `<table><thead><tr><th>#</th><th>Label</th><th>Score</th><th>Box (x1,y1,x2,y2)</th></tr></thead>
     <tbody>${rows}</tbody></table>`;
}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
