"""Interactive book-detection tuning UI (Grounding DINO + SAM masks).

Drag/drop an image, tune the sliders, hit Generate.

Run:
    conda activate makermods
    python grasp/webapp/app.py            # then open http://127.0.0.1:8009
    python grasp/webapp/app.py --port 8010
"""
import argparse
import base64
import io
import time

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

GDINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODELS = {
    "sam2-large":     "facebook/sam2-hiera-large",
    "sam2-base-plus": "facebook/sam2-hiera-base-plus",
    "sam2-small":     "facebook/sam2-hiera-small",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GDINO/SAM: no MPS

_PALETTE = [
    (66, 135, 245), (245, 130, 48), (60, 200, 90), (240, 50, 230),
    (255, 225, 25), (70, 240, 240), (230, 25, 75), (170, 110, 40),
    (145, 30, 180), (0, 200, 200),
]

# ---- lazily-cached models -------------------------------------------------
_cache = {}


def gdino():
    if "gdino" not in _cache:
        proc = AutoProcessor.from_pretrained(GDINO_MODEL)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(DEVICE)
        _cache["gdino"] = (proc, model)
    return _cache["gdino"]


def get_sam(model_key):
    key = f"sam_{model_key}"
    if key not in _cache:
        model_id = SAM_MODELS.get(model_key, SAM_MODELS["sam2-large"])
        proc = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id).to(DEVICE)
        _cache[key] = (proc, model)
    return _cache[key]


# ---- inference helpers ----------------------------------------------------
def gdino_boxes(image_pil, prompt, box_thr, text_thr):
    proc, model = gdino()
    text = prompt.lower().strip()
    if not text.endswith("."):
        text += "."
    inputs = proc(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    return res["boxes"].float().cpu(), res["scores"].float().cpu()


def sam_masks(image_pil, boxes, model_key):
    if len(boxes) == 0:
        return np.zeros((0,) + image_pil.size[::-1], dtype=bool)
    proc, model = get_sam(model_key)
    inputs = proc(image_pil, input_boxes=[boxes.tolist()], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    # out.pred_masks: [1, N, 1, 256, 256] — low-res logits, one per box
    raw = out.pred_masks[0, :, 0]  # [N, 256, 256]
    h, w = image_pil.size[1], image_pil.size[0]
    # upsample each mask to original image size and threshold
    import torch.nn.functional as F
    upsampled = F.interpolate(raw.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False)
    return (upsampled[:, 0] > 0).numpy().astype(bool)  # [N, H, W]


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def dedup_masks(masks, scores, mask_iou_thr, containment):
    order = np.argsort(scores)[::-1]
    areas = masks.reshape(len(masks), -1).sum(1).astype(float)
    keep = []
    for i in order:
        ok = True
        for j in keep:
            if mask_iou(masks[i], masks[j]) > mask_iou_thr:
                ok = False
                break
            if containment:
                inter = np.logical_and(masks[i], masks[j]).sum()
                if areas[i] > 0 and inter / areas[i] > 0.85:
                    ok = False
                    break
        if ok:
            keep.append(i)
    return keep


def book_from_mask(mask):
    m = mask.astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pts = cv2.findNonZero(m)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    obb = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(int).tolist()
    return {
        "box": [x1, y1, x2, y2],
        "obb": obb,
        "centroid": [round(float(cx), 1), round(float(cy), 1)],
        "angle_deg": round(float(angle), 1),
        "area_px": int(m.sum()),
    }


# ---- request handling -----------------------------------------------------
def decode_image(data_url):
    b64 = data_url.split(",", 1)[-1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img


def encode_image(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()



def run_masks(image_pil, p):
    boxes, scores = gdino_boxes(image_pil, p.prompt, p.box_thr, p.text_thr)
    n_raw = len(boxes)
    masks = sam_masks(image_pil, boxes, p.sam_model)
    h, w = image_pil.size[1], image_pil.size[0]
    if len(masks):
        areas = masks.reshape(len(masks), -1).sum(1)
        big = [i for i in range(len(masks)) if areas[i] >= p.min_area * h * w]
        masks, sc = masks[big], scores.numpy()[big]
    else:
        sc = np.array([])
    if p.dedup and len(masks):
        keep = dedup_masks(masks, sc, p.mask_iou, p.containment)
    else:
        keep = list(range(len(masks)))

    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    dets = []
    for rank, i in enumerate(keep):
        info = book_from_mask(masks[i])
        if info is None:
            continue
        info["score"] = round(float(sc[i]), 3)
        color = _PALETTE[rank % len(_PALETTE)]
        overlay[masks[i]] = color
        cv2.polylines(bgr, [np.array(info["obb"])], True, color, 2, cv2.LINE_AA)
        cx, cy = map(int, info["centroid"])
        cv2.circle(bgr, (cx, cy), 4, color, -1)
        cv2.putText(bgr, f"{rank} {info['score']:.2f} {info['angle_deg']:.0f}d",
                    (info["box"][0], max(0, info["box"][1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        dets.append(info)
    bgr = cv2.addWeighted(overlay, 0.4, bgr, 0.6, 0)
    return bgr, dets, n_raw


def sam_auto_masks(image_pil, model_key, grid=12, iou_thr=0.0):
    """Run SAM2 with a point grid over the whole image — no box prompts."""
    import torch.nn.functional as F
    proc, model = get_sam(model_key)
    w, h = image_pil.size
    # build grid points
    xs = np.linspace(w // (grid + 1), w - w // (grid + 1), grid, dtype=int)
    ys = np.linspace(h // (grid + 1), h - h // (grid + 1), grid, dtype=int)
    points = [[int(x), int(y)] for y in ys for x in xs]
    # SAM2 needs 4 levels: [image, object, point, coords] — each point is its own object
    input_points = [[[p] for p in points]]   # [1, N, 1, 2]
    input_labels = [[[1] for _ in points]]   # [1, N, 1]
    inputs = proc(image_pil, input_points=input_points, input_labels=input_labels,
                  return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    raw = out.pred_masks[0, :, 0]  # [N, 256, 256]
    upsampled = F.interpolate(raw.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False)
    masks = (upsampled[:, 0] > 0).numpy().astype(bool)
    scores = out.iou_scores[0, :, 0].cpu().numpy()
    # filter by predicted IoU threshold
    valid = scores >= iou_thr
    return masks[valid], scores[valid]


def run_sam_raw(image_pil, p):
    """Show all SAM masks from a full-image point grid — no GDINO, no dedup."""
    masks, scores = sam_auto_masks(image_pil, p.sam_model, grid=p.grid_size, iou_thr=p.iou_thr)
    n_raw = len(masks)
    # dedup by mask-IoU to remove near-duplicates from overlapping grid points
    keep = dedup_masks(masks, scores, mask_iou_thr=0.5, containment=False) if len(masks) else []
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    dets = []
    for rank, i in enumerate(keep):
        mask = masks[i]
        color = _PALETTE[rank % len(_PALETTE)]
        overlay[mask] = color
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.putText(bgr, f"{rank}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        dets.append({"box": [x1, y1, x2, y2], "score": round(float(scores[i]), 3),
                     "area_px": int(mask.sum())})
    bgr = cv2.addWeighted(overlay, 0.4, bgr, 0.6, 0)
    return bgr, dets, n_raw


def run_boxes(image_pil, p):
    boxes, scores = gdino_boxes(image_pil, p.prompt, p.box_thr, p.text_thr)
    n_raw = len(boxes)
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    dets = []
    for i, (b, s) in enumerate(zip(boxes.tolist(), scores.tolist())):
        x1, y1, x2, y2 = map(int, b)
        color = _PALETTE[i % len(_PALETTE)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{i} {s:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        dets.append({"box": [x1, y1, x2, y2], "score": round(s, 3)})
    return bgr, dets, n_raw


class Params(BaseModel):
    image: str
    mode: str = "boxes"
    sam_model: str = "sam2-large"
    prompt: str = "book."
    box_thr: float = 0.18
    text_thr: float = 0.15
    mask_iou: float = 0.6
    min_area: float = 0.002
    containment: bool = True
    dedup: bool = True
    iou_thr: float = 0.5
    grid_size: int = 12


app = FastAPI(title="book detection tuner")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/detect")
def detect(p: Params):
    t0 = time.time()
    image = decode_image(p.image)
    if p.mode == "boxes":
        bgr, dets, n_raw = run_boxes(image, p)
    elif p.mode == "sam_raw":
        bgr, dets, n_raw = run_sam_raw(image, p)
    else:
        bgr, dets, n_raw = run_masks(image, p)
    return JSONResponse({
        "image": encode_image(bgr),
        "detections": dets,
        "n_raw": int(n_raw),
        "n_final": len(dets),
        "seconds": round(time.time() - t0, 2),
        "device": DEVICE,
    })


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Book Detection Tuner</title>
<style>
  :root{--bg:#0f1117;--panel:#181b24;--ink:#e6e8ee;--mut:#8b93a7;--acc:#4f8cff;--line:#262b38}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
  header{padding:14px 20px;border-bottom:1px solid var(--line);font-weight:600}
  .wrap{display:grid;grid-template-columns:320px 1fr;gap:18px;padding:18px;align-items:start}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
  h3{margin:0 0 12px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
  label{display:block;margin:14px 0 4px;color:var(--mut);font-size:12px}
  input[type=text]{width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:#11141c;color:var(--ink)}
  input[type=range]{width:100%}
  .row{display:flex;justify-content:space-between;align-items:center}
  .val{font-variant-numeric:tabular-nums;color:var(--ink);font-size:12px}
  .tabs{display:flex;gap:8px;margin-bottom:14px}
  .tabs button{flex:1;padding:8px;border-radius:8px;border:1px solid var(--line);background:#11141c;color:var(--mut);cursor:pointer;font-size:13px}
  .tabs button.on{border-color:var(--acc);color:#fff;background:#16203a}
  .dim{opacity:.35;pointer-events:none}
  #drop{border:2px dashed var(--line);border-radius:12px;padding:26px;text-align:center;color:var(--mut);cursor:pointer}
  #drop.hot{border-color:var(--acc);color:var(--ink)}
  #go{margin-top:16px;width:100%;padding:11px;border:0;border-radius:10px;background:var(--acc);color:#fff;font-weight:600;cursor:pointer}
  #go:disabled{opacity:.5;cursor:not-allowed}
  .hint{color:var(--mut);font-size:11px;margin-top:6px}
  .out{min-height:300px;display:flex;align-items:center;justify-content:center}
  .out img{max-width:100%;border-radius:10px}
  .stat{display:flex;gap:18px;color:var(--mut);font-size:12px;margin:10px 2px}
  .stat b{color:var(--ink)}
  table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
  th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
  th{color:var(--mut);font-weight:500}
</style></head>
<body>
<header>Book Detection Tuner &mdash; Grounding DINO + SAM 2</header>
<div class="wrap">
  <div class="panel">
    <h3>1. Image</h3>
    <div id="drop">Drag &amp; drop an image<br>or click to choose</div>
    <input id="file" type="file" accept="image/*" hidden>

    <h3 style="margin-top:18px">2. Mode</h3>
    <div class="tabs">
      <button id="m-boxes" class="on">DINO Boxes</button>
      <button id="m-sam-raw">SAM Raw</button>
      <button id="m-masks">SAM + Dedup</button>
    </div>

    <h3>3. Settings</h3>

    <label>Prompt</label>
    <input id="prompt" type="text" value="book.">

    <label>Box threshold <span class="val" id="v-box">0.18</span></label>
    <input id="box_thr" type="range" min="0" max="0.6" step="0.01" value="0.18">

    <label>Text threshold <span class="val" id="v-text">0.15</span></label>
    <input id="text_thr" type="range" min="0" max="0.6" step="0.01" value="0.15">

    <div id="raw-settings">
      <label>Grid size <span class="val" id="v-grid">12</span></label>
      <input id="grid_size" type="range" min="4" max="24" step="1" value="12">

      <label>Min IoU score <span class="val" id="v-iou">0.50</span></label>
      <input id="iou_thr" type="range" min="0" max="1" step="0.01" value="0.5">
    </div>

    <div id="sam-settings">
      <label>SAM 2 model</label>
      <select id="sam_model" style="width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:#11141c;color:var(--ink)">
        <option value="sam2-large">SAM 2 Large (best)</option>
        <option value="sam2-base-plus">SAM 2 Base+</option>
        <option value="sam2-small">SAM 2 Small (fastest)</option>
      </select>

      <label>Mask IoU dedup <span class="val" id="v-mask">0.60</span></label>
      <input id="mask_iou" type="range" min="0" max="1" step="0.01" value="0.6">

      <label>Min area (% of image) <span class="val" id="v-area">0.20</span></label>
      <input id="min_area" type="range" min="0" max="0.03" step="0.001" value="0.002">

      <label class="row" style="margin-top:14px"><span>Dedup masks (mask IoU NMS)</span>
        <input id="dedup" type="checkbox" checked></label>

      <label class="row" style="margin-top:8px"><span>Suppress contained masks</span>
        <input id="containment" type="checkbox" checked></label>
    </div>

    <button id="go" disabled>Generate</button>
    <div class="hint" id="hint">Upload an image to start. First run loads the models (slow); later runs are fast.</div>
  </div>

  <div class="panel">
    <h3>Result</h3>
    <div class="out" id="out"><span style="color:var(--mut)">No result yet.</span></div>
    <div class="stat" id="stat" style="display:none">
      <span>raw: <b id="s-raw">-</b></span>
      <span>final: <b id="s-final">-</b></span>
      <span>time: <b id="s-time">-</b>s</span>
      <span>device: <b id="s-dev">-</b></span>
    </div>
    <table id="tbl" style="display:none"><thead><tr id="thead"></tr></thead><tbody id="tbody"></tbody></table>
  </div>
</div>
<script>
let imageData=null, mode="boxes";
const $=id=>document.getElementById(id);
const drop=$("drop"), file=$("file"), go=$("go");

const links={box_thr:"v-box",text_thr:"v-text",mask_iou:"v-mask",min_area:"v-area",grid_size:"v-grid",iou_thr:"v-iou"};
for(const [id,lab] of Object.entries(links)){
  const f=()=>{ let v=parseFloat($(id).value);
    if(id==="min_area") $(lab).textContent=(v*100).toFixed(2);
    else if(id==="grid_size") $(lab).textContent=v.toFixed(0);
    else $(lab).textContent=v.toFixed(2); };
  $(id).addEventListener("input",f); f();
}

drop.onclick=()=>file.click();
file.onchange=e=>load(e.target.files[0]);
["dragover","dragenter"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("hot")}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("hot")}));
drop.addEventListener("drop",e=>load(e.dataTransfer.files[0]));
function load(f){ if(!f)return; const r=new FileReader();
  r.onload=()=>{imageData=r.result; go.disabled=false; drop.innerHTML="";
    const im=new Image(); im.src=imageData; im.style.maxWidth="100%"; im.style.borderRadius="8px"; drop.appendChild(im);
    $("out").innerHTML='<span style="color:var(--mut)">Ready. Click Generate.</span>'; };
  r.readAsDataURL(f);
}

function setMode(m){ mode=m;
  $("m-boxes").classList.toggle("on", m==="boxes");
  $("m-sam-raw").classList.toggle("on", m==="sam_raw");
  $("m-masks").classList.toggle("on", m==="masks");
  $("raw-settings").classList.toggle("dim", m!=="sam_raw");
  $("sam-settings").classList.toggle("dim", m!=="masks");
}
$("m-boxes").onclick=()=>setMode("boxes");
$("m-sam-raw").onclick=()=>setMode("sam_raw");
$("m-masks").onclick=()=>setMode("masks");
setMode("boxes");

go.onclick=async()=>{
  if(!imageData)return;
  go.disabled=true; const t=go.textContent; go.textContent="Running…";
  $("hint").textContent="Running inference… (first call loads models, can take a minute)";
  try{
    const body={image:imageData, mode,
      sam_model:$("sam_model").value, prompt:$("prompt").value,
      box_thr:+$("box_thr").value, text_thr:+$("text_thr").value,
      mask_iou:+$("mask_iou").value, min_area:+$("min_area").value,
      containment:$("containment").checked, dedup:$("dedup").checked,
      iou_thr:+$("iou_thr").value, grid_size:+$("grid_size").value};
    const r=await fetch("/api/detect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){ $("out").innerHTML='<span style="color:#f66">'+d.error+'</span>'; throw new Error(d.error); }
    $("out").innerHTML=""; const im=new Image(); im.src=d.image; $("out").appendChild(im);
    $("stat").style.display="flex";
    $("s-raw").textContent=d.n_raw; $("s-final").textContent=d.n_final;
    $("s-time").textContent=d.seconds; $("s-dev").textContent=d.device;
    renderTable(d.detections);
    $("hint").textContent="Done. Tweak sliders and Generate again.";
  }catch(e){ $("out").innerHTML='<span style="color:#f66">Error: '+e+'</span>'; }
  go.disabled=false; go.textContent=t;
};

function renderTable(dets){

  const tbl=$("tbl"), th=$("thead"), tb=$("tbody");
  if(!dets||!dets.length){tbl.style.display="none";return;}
  const cols = ("angle_deg" in dets[0]) ? ["#","score","box","angle_deg","centroid","area_px"] : ["#","score","box"];
  th.innerHTML=cols.map(c=>`<th>${c}</th>`).join("");
  tb.innerHTML=dets.map((d,i)=>"<tr>"+cols.map(c=>{
    if(c==="#")return `<td>${i}</td>`;
    let v=d[c]; if(Array.isArray(v))v="["+v.join(", ")+"]";
    return `<td>${v??"-"}</td>`;
  }).join("")+"</tr>").join("");
  tbl.style.display="table";
}
</script>
</body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8009)
    args = ap.parse_args()
    print(f"book detection tuner -> http://{args.host}:{args.port}  (device={DEVICE})")
    uvicorn.run(app, host=args.host, port=args.port)
