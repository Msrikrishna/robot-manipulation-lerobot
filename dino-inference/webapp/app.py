"""DINOv3 playground backend.

A small FastAPI server that wraps the two inference pipelines in this repo
(PCA feature maps + monocular depth) behind a web UI. Models are loaded lazily
and cached so repeated requests are fast.

Run:
    uv run uvicorn webapp.app:app --reload --port 8000
    # then open http://localhost:8000

Endpoints:
    GET  /            -> the single-page UI (index.html)
    GET  /api/config  -> available models / defaults for the UI
    POST /api/pca     -> multipart image + params, returns base64 PNG + stats
    POST /api/depth   -> multipart image + params, returns base64 PNG + stats
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import multiprocessing as mp
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

# Quiet the depth_anything_3 package's stdout logger (read at its import time).
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

HERE = Path(__file__).parent

# ---- model registry -------------------------------------------------------
# Backbones usable for the PCA feature map. Smaller = faster on the M2 Pro.
PCA_MODELS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

# Depth heads selectable in the UI. `metric` flags models that output depth in
# real meters (larger = farther); the rest are relative inverse depth
# (larger = nearer). `gated` models need a HF license accept + login first.
DEPTH_MODELS = {
    "dinov3-chmv2": {
        "repo": "facebook/dinov3-vitl16-chmv2-dpt-head",
        "label": "DINOv3 ViT-L · CHMv2 (canopy / aerial)",
        "desc": "Satellite canopy-height head. Top-down imagery only — bad for "
                "ground-level scenes.",
        "gated": True, "metric": True, "size_mb": 1348,
        "backbone": "DINOv3 ViT-L/16", "dataset": "LiDAR canopy maps (satellite)",
    },
    "dinov2-nyu-small": {
        "repo": "facebook/dpt-dinov2-small-nyu",
        "label": "DPT · DINOv2-S · NYU (indoor)",
        "desc": "Lightest indoor depth head. Good first pick for tabletop scenes.",
        "gated": False, "metric": True, "size_mb": 149,
        "backbone": "DINOv2-S", "dataset": "NYU Depth v2 (indoor)",
    },
    "dinov2-nyu-base": {
        "repo": "facebook/dpt-dinov2-base-nyu",
        "label": "DPT · DINOv2-B · NYU (indoor)",
        "desc": "Indoor (NYU Depth v2) metric depth. Balanced speed/quality.",
        "gated": False, "metric": True, "size_mb": 448,
        "backbone": "DINOv2-B", "dataset": "NYU Depth v2 (indoor)",
    },
    "dinov2-nyu-giant": {
        "repo": "facebook/dpt-dinov2-giant-nyu",
        "label": "DPT · DINOv2-G · NYU (indoor)",
        "desc": "Heaviest, sharpest indoor head. Slow on CPU/MPS; large download.",
        "gated": False, "metric": True, "size_mb": 9596,
        "backbone": "DINOv2-G", "dataset": "NYU Depth v2 (indoor)",
    },
    "depth-anything-v2-small": {
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "label": "Depth Anything V2 · Small (relative)",
        "desc": "Smallest credible depth model. Relative inverse depth; great "
                "quality-per-MB for quick previews.",
        "gated": False, "metric": False, "size_mb": 99,
        "backbone": "DINOv2-S", "dataset": "synthetic + 62M pseudo-labeled real",
    },
    "depth-anything-v2-base": {
        "repo": "depth-anything/Depth-Anything-V2-Base-hf",
        "label": "Depth Anything V2 · Base (relative)",
        "desc": "Mid-size DA-V2. Relative inverse depth; better detail than Small.",
        "gated": False, "metric": False, "size_mb": 390,
        "backbone": "DINOv2-B", "dataset": "synthetic + 62M pseudo-labeled real",
    },
    "depth-anything-v2-large": {
        "repo": "depth-anything/Depth-Anything-V2-Large-hf",
        "label": "Depth Anything V2 · Large (relative)",
        "desc": "Best all-round monocular depth. Relative inverse depth, sharp edges.",
        "gated": False, "metric": False, "size_mb": 1341,
        "backbone": "DINOv2-L", "dataset": "synthetic + 62M pseudo-labeled real",
    },
    "depth-anything-v2-metric-indoor-small": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "label": "Depth Anything V2 · Metric Indoor (Small)",
        "desc": "Smallest metric indoor model — metric meters at 99 MB. Best "
                "lightweight pick for manipulation.",
        "gated": False, "metric": True, "size_mb": 99,
        "backbone": "DINOv2-S", "dataset": "Hypersim (indoor)",
    },
    "depth-anything-v2-metric-indoor-base": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "label": "Depth Anything V2 · Metric Indoor (Base)",
        "desc": "Mid-size metric indoor. Better detail than Small.",
        "gated": False, "metric": True, "size_mb": 390,
        "backbone": "DINOv2-B", "dataset": "Hypersim (indoor)",
    },
    "depth-anything-v2-metric-indoor": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "label": "Depth Anything V2 · Metric Indoor (Large)",
        "desc": "Metric meters, indoor (Hypersim). Sharpest metric indoor model.",
        "gated": False, "metric": True, "size_mb": 1341,
        "backbone": "DINOv2-L", "dataset": "Hypersim (indoor)",
    },
    "depth-anything-v2-metric-outdoor-small": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "label": "Depth Anything V2 · Metric Outdoor (Small)",
        "desc": "Smallest metric outdoor model — metric meters at 99 MB.",
        "gated": False, "metric": True, "size_mb": 99,
        "backbone": "DINOv2-S", "dataset": "Virtual KITTI (outdoor)",
    },
    "depth-anything-v2-metric-outdoor-base": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "label": "Depth Anything V2 · Metric Outdoor (Base)",
        "desc": "Mid-size metric outdoor. Better detail than Small.",
        "gated": False, "metric": True, "size_mb": 390,
        "backbone": "DINOv2-B", "dataset": "Virtual KITTI (outdoor)",
    },
    "depth-anything-v2-metric-outdoor": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "label": "Depth Anything V2 · Metric Outdoor (Large)",
        "desc": "Metric meters, outdoor / driving-scale (Virtual KITTI).",
        "gated": False, "metric": True, "size_mb": 1341,
        "backbone": "DINOv2-L", "dataset": "Virtual KITTI (outdoor)",
    },
    # Depth Anything 3 — NOT transformers-compatible; run via the
    # depth_anything_3 package (loader="da3"). Relative depth. Heavy multi-view
    # geometry transformer: ~3s/frame on MPS, so fine for single images but slow
    # for the live tab.
    "da3-small": {
        "repo": "depth-anything/DA3-SMALL",
        "label": "Depth Anything 3 · Small (relative)",
        "desc": "DA3 via depth_anything_3 pkg. Relative depth + camera pose. "
                "~3s/frame on MPS — slow for live.",
        "gated": False, "metric": False, "size_mb": 137,
        "backbone": "DINOv2-S (DA3)", "dataset": "DA3 academic mix", "loader": "da3",
    },
    "da3-base": {
        "repo": "depth-anything/DA3-BASE",
        "label": "Depth Anything 3 · Base (relative)",
        "desc": "DA3 0.1B. Relative depth + pose. Slower than Small; single-image use.",
        "gated": False, "metric": False, "size_mb": 542,
        "backbone": "DINOv2-B (DA3)", "dataset": "DA3 academic mix", "loader": "da3",
    },
    "da3-large": {
        "repo": "depth-anything/DA3-LARGE",
        "label": "Depth Anything 3 · Large (relative)",
        "desc": "DA3 0.4B. Sharpest DA3; relative depth + pose. Single-image use only.",
        "gated": False, "metric": False, "size_mb": 1644,
        "backbone": "DINOv2-L (DA3)", "dataset": "DA3 academic mix", "loader": "da3",
    },
    "da3-metric-large": {
        "repo": "depth-anything/DA3METRIC-LARGE",
        "label": "Depth Anything 3 · Metric Large",
        "desc": "DA3 metric monocular (only metric size in V3 — no small/base). "
                "Canonical metric; true metres need the camera focal length. "
                "~3s/frame — single-image use.",
        "gated": False, "metric": True, "size_mb": 1337,
        "backbone": "DINOv2-L (DA3)", "dataset": "DA3 metric mix", "loader": "da3",
    },
    "da3-mono-large": {
        "repo": "depth-anything/DA3MONO-LARGE",
        "label": "Depth Anything 3 · Mono Large (relative)",
        "desc": "DA3 monocular variant, tuned for single-image (vs the multi-view "
                "default). Relative depth. ~3s/frame.",
        "gated": False, "metric": False, "size_mb": 1337,
        "backbone": "DINOv2-L (DA3)", "dataset": "DA3 academic mix", "loader": "da3",
    },
}

# Default selection. Env override accepts either a registry key or a raw repo id.
_env_depth = os.environ.get("DEPTH_MODEL")
if _env_depth and _env_depth not in DEPTH_MODELS:
    # raw repo id passed in — register it so the UI can show it too
    DEPTH_MODELS[_env_depth] = {
        "repo": _env_depth, "label": _env_depth, "desc": "Custom (DEPTH_MODEL env).",
        "gated": False, "metric": False,
    }
DEFAULT_DEPTH_KEY = _env_depth or "dinov2-nyu-base"

# Model used by the live-camera tab: smallest metric-indoor head (fast on MPS).
LIVE_DEPTH_KEY = "depth-anything-v2-metric-indoor-small"


def depth_repo(key: str) -> str:
    return DEPTH_MODELS.get(key, DEPTH_MODELS[DEFAULT_DEPTH_KEY])["repo"]


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- colormaps ------------------------------------------------------------
_TURBO = np.array(
    [
        [48, 18, 59], [70, 134, 251], [27, 207, 212], [90, 240, 110],
        [220, 220, 40], [250, 152, 40], [220, 60, 30], [122, 4, 3],
    ],
    dtype=np.float32,
)
_MAGMA = np.array(
    [
        [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
        [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191],
    ],
    dtype=np.float32,
)


def apply_colormap(d01: np.ndarray, name: str) -> Image.Image:
    """Map a HxW array in [0,1] to an RGB image via the named colormap."""
    if name == "gray":
        g = (d01 * 255).astype("uint8")
        return Image.fromarray(np.stack([g, g, g], -1))
    stops = _MAGMA if name == "magma" else _TURBO
    idx = d01 * (len(stops) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[hi] * frac
    return Image.fromarray(rgb.astype("uint8"))


# ---- lazy model cache -----------------------------------------------------
@lru_cache(maxsize=4)
def get_backbone(model_id: str):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device()).eval()
    return proc, model


@lru_cache(maxsize=3)
def get_depth_pipe(model_id: str):
    from transformers import pipeline

    return pipeline("depth-estimation", model=model_id, device=device())


@lru_cache(maxsize=1)
def get_da3_model(repo: str):
    """Lazy-load a Depth Anything 3 model via the depth_anything_3 package.

    DA3 is not transformers-compatible, so it can't use the pipeline path. We
    stub out the package's 3D/video export submodule (pulls moviepy/open3d we
    don't ship) before importing, and silence its stdout logger. maxsize=1 —
    these are large; keep only the most recent in memory.
    """
    import sys
    import types

    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
    if "depth_anything_3.utils.export" not in sys.modules:
        stub = types.ModuleType("depth_anything_3.utils.export")
        stub.export = lambda *a, **k: None
        sys.modules["depth_anything_3.utils.export"] = stub
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(repo).to(device()).eval()


def _da3_depth(img: Image.Image, repo: str) -> np.ndarray:
    """Run a DA3 model on one image -> (H, W) float32 depth array."""
    model = get_da3_model(repo)
    with torch.no_grad():
        pred = model.inference([img], export_dir=None)
    d = pred.depth
    d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
    return d[0] if d.ndim == 3 else d


# ---- model download + cache tracking --------------------------------------
# Inference files we need; HF only downloads the patterns a repo actually has.
_DL_PATTERNS = ["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"]


def _repo_snapshot_dir(repo: str) -> Path | None:
    """The cache `snapshots/<rev>/` dir for a repo (entries are symlinks into
    `blobs/`), or None if nothing is cached."""
    from huggingface_hub.constants import HF_HUB_CACHE

    base = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not base.exists():
        return None
    revs = [d for d in base.iterdir() if d.is_dir()]
    return revs[0] if revs else None


def _is_cached(repo: str) -> bool:
    """True only if the repo is *fully* downloaded — i.e. a weights file is
    present and there are no partial `*.incomplete` files. Checking config.json
    alone is a false positive: it downloads first, so a half-finished (or merely
    probed) model would wrongly look ready and then block on a weight download
    at inference time."""
    try:
        blobs = _repo_blobs_dir(repo)
        if not blobs.exists():
            return False
        if any(p.name.endswith(".incomplete") for p in blobs.iterdir()):
            return False  # a download is in progress / was interrupted
        snap = _repo_snapshot_dir(repo)
        if snap is None:
            return False
        return any(
            f.suffix in (".safetensors", ".bin") and f.exists()
            for f in snap.rglob("*")
        )
    except Exception:
        return False


def _repo_blobs_dir(repo: str) -> Path:
    """The cache `blobs/` dir where the real downloaded files (and partial
    `*.incomplete` files) for a repo live."""
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = "models--" + repo.replace("/", "--")
    return Path(HF_HUB_CACHE) / folder / "blobs"


def _downloaded_bytes(repo: str) -> int:
    """Bytes on disk for a repo so far. Only the blobs dir is summed (snapshot
    entries are symlinks into it, so this avoids double counting) and it
    includes in-progress `*.incomplete` temp files."""
    blobs = _repo_blobs_dir(repo)
    if not blobs.exists():
        return 0
    total = 0
    for f in blobs.iterdir():
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _repo_total_bytes(repo: str) -> int:
    """Total download size from the HF API, summed over the files we fetch.
    Returns 0 if the API can't be reached (UI then shows an indeterminate bar)."""
    import fnmatch

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        total = 0
        for s in info.siblings:
            name = s.rfilename
            if s.size and any(fnmatch.fnmatch(name, p) for p in _DL_PATTERNS):
                total += s.size
        return total
    except Exception:
        return 0


# A download runs in its own (spawned) process so it can be cancelled when the
# user navigates away. Jobs are keyed by repo and shared across SSE clients, so
# two browser tabs asking for the same model don't kick off duplicate
# downloads. `_DL_LOCK` guards the registry.
_MP_CTX = mp.get_context("spawn")
_DL_LOCK = threading.Lock()
_DL_JOBS: dict[str, "_DLJob"] = {}


class _DLJob:
    def __init__(self, repo: str):
        self.repo = repo
        self.total = _repo_total_bytes(repo)
        self.success = False
        self.error: str | None = None
        self.clients = 0
        self._q = _MP_CTX.Queue()
        self._proc = _MP_CTX.Process(
            target=_run_download, args=(repo, _DL_PATTERNS, self._q), daemon=True
        )

    def start(self):
        self._proc.start()

    def _drain(self):
        try:
            while not self._q.empty():
                self.success, self.error = self._q.get_nowait()
        except Exception:
            pass

    @property
    def done(self) -> bool:
        self._drain()
        return self._proc.exitcode is not None

    def stop(self):
        if self._proc.is_alive():
            self._proc.terminate()


def _run_download(repo: str, patterns: list[str], q) -> None:
    # Indirection so the spawned process imports the tiny _downloader module
    # (huggingface_hub only) rather than this one (torch/transformers).
    from webapp._downloader import download

    download(repo, patterns, q)


def _acquire_job(repo: str) -> _DLJob:
    """Get the live download job for a repo (starting one if needed) and
    register this caller as a viewer."""
    with _DL_LOCK:
        job = _DL_JOBS.get(repo)
        if job is None or (job.done and not job.success):
            job = _DLJob(repo)
            _DL_JOBS[repo] = job
            job.start()
        job.clients += 1
        return job


def _release_job(repo: str):
    """Drop a viewer. If the last one leaves before the download finishes,
    cancel it — HF leaves a `*.incomplete` file so a later attempt resumes."""
    with _DL_LOCK:
        job = _DL_JOBS.get(repo)
        if job is None:
            return
        job.clients -= 1
        if job.done:
            _DL_JOBS.pop(repo, None)
        elif job.clients <= 0:
            job.stop()
            _DL_JOBS.pop(repo, None)


# ---- helpers --------------------------------------------------------------
# Hard ceiling so an enormous upload can never blow up memory/compute even if
# the client asks for more. Depth runs at the (capped) native resolution.
MAX_SIDE_CEILING = 1536


def read_image(upload_bytes: bytes, max_side: int = 1024) -> Image.Image:
    """Decode and downscale so the longest side is <= max_side (never upscale).

    This is the key cost control: every later step (model forward, output
    resize, PNG encode) is bounded by this, not by the raw upload size.
    """
    img = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
    cap = min(int(max_side), MAX_SIDE_CEILING)
    longest = max(img.size)
    if longest > cap:
        scale = cap / longest
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.BICUBIC,
        )
    return img


def to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def side_by_side(a: Image.Image, b: Image.Image, gap: int = 10) -> Image.Image:
    canvas = Image.new("RGB", (a.width + gap + b.width, max(a.height, b.height)), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    return canvas


# ---- pipelines ------------------------------------------------------------
def run_pca(
    img: Image.Image,
    model_id: str,
    size: int,
    pct_low: float,
    pct_high: float,
    n_components: int = 3,
    remove_bg: bool = False,
    bg_threshold: float = 0.0,
):
    proc, model = get_backbone(model_id)
    patch = model.config.patch_size
    w, h = img.size
    scale = size / max(w, h)
    nw = max(patch, round(w * scale / patch) * patch)
    nh = max(patch, round(h * scale / patch) * patch)
    img_in = img.resize((nw, nh), Image.BICUBIC)
    inputs = proc(
        images=img_in, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to(device())

    with torch.no_grad():
        out = model(**inputs)
    tokens = out.last_hidden_state[0]

    _, _, H, W = inputs["pixel_values"].shape
    gh, gw = H // patch, W // patch
    n_patches = gh * gw
    patch_tokens = tokens[-n_patches:].float().cpu()

    feats = patch_tokens - patch_tokens.mean(0, keepdim=True)
    q = max(3, n_components)
    _, _, V = torch.pca_lowrank(feats, q=q)
    proj_full = feats @ V[:, :q]  # (n_patches, q)

    fg_mask = None
    if remove_bg:
        # First PCA component often separates fg/bg. Threshold it (normalized).
        c0 = proj_full[:, 0].numpy()
        c0n = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-8)
        fg_mask = c0n > bg_threshold

    proj = proj_full[:, :3].reshape(gh, gw, 3).numpy()
    lo = np.percentile(proj, pct_low, axis=(0, 1))
    hi = np.percentile(proj, pct_high, axis=(0, 1))
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    if fg_mask is not None:
        m = fg_mask.reshape(gh, gw)[..., None]
        rgb = rgb * m  # black-out background patches
    pca_img = Image.fromarray((rgb * 255).astype("uint8")).resize(img.size, Image.NEAREST)
    info = {
        "grid": f"{gh}x{gw}",
        "patches": int(n_patches),
        "dim": int(patch_tokens.shape[1]),
        "input_size": f"{nw}x{nh}",
        "patch_size": int(patch),
    }
    return pca_img, info


def run_depth(
    img: Image.Image,
    colormap: str,
    model_key: str,
    pct_low: float = 2.0,
    pct_high: float = 98.0,
    near: float = 0.0,
    far: float = 0.0,
    gamma: float = 1.0,
):
    """Predict depth and map it to colour.

    The display range — not just min/max — decides how much near-vs-far contrast
    you see. A few far pixels with plain min/max normalization crush the whole
    foreground into one colour. So we stretch the colormap over a *clipped*
    range: an explicit [near, far] in metres if given, else a robust
    [pct_low, pct_high] percentile window. `gamma` (<1 = more near contrast)
    further reshapes the ramp.
    """
    spec = DEPTH_MODELS.get(model_key, DEPTH_MODELS[DEFAULT_DEPTH_KEY])
    repo = spec["repo"]
    metric = spec.get("metric", False)
    if spec.get("loader") == "da3":
        depth = _da3_depth(img, repo)
    else:
        pipe = get_depth_pipe(repo)
        depth = pipe(img)["predicted_depth"].squeeze().float().cpu().numpy()
    depth_img, info = render_depth(
        depth, metric, repo, img.size, colormap, pct_low, pct_high, near, far, gamma
    )
    # depth + metric are returned too so the caller can cache them and re-colour
    # (gamma/range/colormap) live without re-running the model.
    return depth_img, info, depth, metric


def render_depth(depth, metric, repo, size, colormap, pct_low, pct_high, near, far, gamma):
    """Pure colour-mapping step: depth array (+ display params) -> colour image.
    No model involved, so this is cheap enough to call on every slider move."""
    raw_min, raw_max = float(depth.min()), float(depth.max())
    # Pick the display window: manual metres take priority, else percentiles.
    if near > 0 or far > 0:
        lo = near if near > 0 else raw_min
        hi = far if far > 0 else raw_max
    else:
        lo = float(np.percentile(depth, pct_low))
        hi = float(np.percentile(depth, pct_high))
    if hi <= lo:
        hi = lo + 1e-6

    d01 = np.clip((depth - lo) / (hi - lo), 0, 1)
    if gamma and gamma != 1.0:
        d01 = d01 ** float(gamma)
    # Metric models output distance (larger = farther); invert so the colormap's
    # "warmer = nearer" convention holds across both model families.
    if metric:
        d01 = 1.0 - d01
    depth_img = apply_colormap(d01, colormap).resize(size, Image.BILINEAR)

    span = hi - lo
    info = {
        "shape": f"{depth.shape[0]}x{depth.shape[1]}",
        "raw_min": round(raw_min, 3),
        "raw_max": round(raw_max, 3),
        "shown_range": f"{lo:.2f}–{hi:.2f} {'m' if metric else 'rel'}",
        "units": "meters" if metric else "relative",
        "model": repo,
    }
    if metric:
        # cm of depth per 8-bit colour step over the shown range = display
        # granularity. Narrower range -> finer steps -> more visible relief.
        info["cm_per_color_step"] = round(span / 255 * 100, 2)
    return depth_img, info


# Cache of recent raw depth arrays so the colour controls (gamma/range/colormap)
# can re-render without re-running the model. Bounded; oldest evicted.
from collections import OrderedDict  # noqa: E402

_DEPTH_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_DEPTH_CACHE_MAX = 8
_DEPTH_CACHE_LOCK = threading.Lock()

# Similarity cache: token -> {sim_map (gh x gw float32), clicked (ci, cj), grid, orig}
_SIM_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_SIM_CACHE_MAX = 8
_SIM_CACHE_LOCK = threading.Lock()


def _cache_sim(sim_map: np.ndarray, clicked: tuple, grid: tuple, orig: Image.Image) -> str:
    import uuid
    token = uuid.uuid4().hex
    with _SIM_CACHE_LOCK:
        _SIM_CACHE[token] = {"sim_map": sim_map, "clicked": clicked, "grid": grid, "orig": orig}
        while len(_SIM_CACHE) > _SIM_CACHE_MAX:
            _SIM_CACHE.popitem(last=False)
    return token


def _cache_depth(depth, metric: bool, repo: str, orig: Image.Image) -> str:
    import uuid

    token = uuid.uuid4().hex
    with _DEPTH_CACHE_LOCK:
        _DEPTH_CACHE[token] = {"depth": depth, "metric": metric, "repo": repo, "orig": orig}
        while len(_DEPTH_CACHE) > _DEPTH_CACHE_MAX:
            _DEPTH_CACHE.popitem(last=False)
    return token


def _collect_4d(obj, out, depth=2):
    """Recursively gather float (B,C,H,W) tensors from nested lists/tuples."""
    if torch.is_tensor(obj):
        if obj.dim() == 4 and obj.is_floating_point():
            out.append(obj)
    elif depth > 0 and isinstance(obj, (list, tuple)):
        for x in obj:
            _collect_4d(x, out, depth - 1)


def _backbone_patch_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, "patch_size"):
        return int(cfg.patch_size)
    bb = getattr(cfg, "backbone_config", None)
    if bb is not None and hasattr(bb, "patch_size"):
        return int(bb.patch_size)
    return 16


def run_depth_pca(img: Image.Image, model_key: str, size: int, pct_low: float, pct_high: float):
    """PCA-visualize a depth model's internal embeddings. Prefers the fused
    feature map feeding the depth head (depth-specialized); falls back to the
    backbone's last hidden state when the head input isn't a plain tensor."""
    spec = DEPTH_MODELS.get(model_key, DEPTH_MODELS[DEFAULT_DEPTH_KEY])
    repo = spec["repo"]
    pipe = get_depth_pipe(repo)
    model, proc, dev = pipe.model, pipe.image_processor, device()

    work = img
    longest = max(img.size)
    if longest > size:  # bound cost before the processor's own resize
        s = size / longest
        work = img.resize(
            (max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC
        )
    inputs = proc(images=work, return_tensors="pt").to(dev)

    captured: list = []
    head = getattr(model, "head", None)
    handle = None
    if head is not None:
        def _hook(mod, args, kwargs):
            for a in args:
                _collect_4d(a, captured)
            for v in kwargs.values():
                _collect_4d(v, captured)
        handle = head.register_forward_pre_hook(_hook, with_kwargs=True)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    if handle is not None:
        handle.remove()

    if captured:
        feat = max(captured, key=lambda t: t.shape[-1] * t.shape[-2])[0].float().cpu()
        c, gh, gw = feat.shape
        feats = feat.reshape(c, gh * gw).permute(1, 0)  # (n_patches, C)
        source, dim = "fused head features", c
    else:
        tokens = out.hidden_states[-1][0].float().cpu()  # (N, D)
        _, _, H, W = inputs["pixel_values"].shape
        patch = _backbone_patch_size(model)
        gh, gw = H // patch, W // patch
        feats = tokens[-(gh * gw):]  # drop leading cls/register tokens
        source, dim = "backbone tokens", int(feats.shape[1])

    feats = feats - feats.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(feats, q=3)
    proj = (feats @ V[:, :3]).reshape(gh, gw, 3).numpy()
    lo = np.percentile(proj, pct_low, axis=(0, 1))
    hi = np.percentile(proj, pct_high, axis=(0, 1))
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    pca_img = Image.fromarray((rgb * 255).astype("uint8")).resize(img.size, Image.NEAREST)
    info = {"source": source, "grid": f"{gh}x{gw}", "dim": int(dim), "model": repo}
    return pca_img, info


# ---- similarity (click-a-patch) -------------------------------------------

def run_similarity_multi(
    img: Image.Image,
    model_id: str,
    size: int,
    points: list,      # list of (cx, cy) fractional coords
    threshold: float,
    colormap: str,
    overlay_alpha: float,
):
    proc, model = get_backbone(model_id)
    patch = model.config.patch_size
    w, h = img.size
    scale = size / max(w, h)
    nw = max(patch, round(w * scale / patch) * patch)
    nh = max(patch, round(h * scale / patch) * patch)
    img_in = img.resize((nw, nh), Image.BICUBIC)
    inputs = proc(
        images=img_in, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to(device())

    with torch.no_grad():
        out = model(**inputs)
    tokens = out.last_hidden_state[0]

    _, _, H, W = inputs["pixel_values"].shape
    gh, gw = H // patch, W // patch
    n_patches = gh * gw
    patch_tokens = tokens[-n_patches:].float().cpu()  # (n_patches, D)

    # Collect query tokens for all clicked points and average them
    query_tokens = []
    clicked_list = []
    for cx, cy in points:
        ci = int(np.clip(cy * gh, 0, gh - 1))
        cj = int(np.clip(cx * gw, 0, gw - 1))
        query_tokens.append(patch_tokens[ci * gw + cj])
        clicked_list.append((ci, cj))

    query = torch.stack(query_tokens).mean(0)  # averaged embedding (D,)

    q_norm = query / (query.norm() + 1e-8)
    t_norm = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + 1e-8)
    sim = (t_norm @ q_norm).numpy()
    sim_grid = sim.reshape(gh, gw).astype(np.float32)

    # Use first clicked point as the representative marker
    token = _cache_sim(sim_grid, clicked_list[0], (gh, gw), img)
    # Store all clicked points in the cache entry for multi-dot rendering
    with _SIM_CACHE_LOCK:
        if token in _SIM_CACHE:
            _SIM_CACHE[token]["clicked_all"] = clicked_list

    result_img, info = render_similarity_multi(
        sim_grid, clicked_list, img, threshold, colormap, overlay_alpha
    )
    info["n_points"] = len(clicked_list)
    return result_img, info, token


def render_similarity_multi(
    sim_grid: np.ndarray,
    clicked_list: list,
    orig: Image.Image,
    threshold: float,
    colormap: str,
    overlay_alpha: float,
) -> tuple[Image.Image, dict]:
    gh, gw = sim_grid.shape

    lo, hi = sim_grid.min(), sim_grid.max()
    sim_01 = (sim_grid - lo) / (hi - lo + 1e-8)
    colored = np.array(apply_colormap(sim_01, colormap).resize(orig.size, Image.BILINEAR), dtype=np.float32)

    mask_patch = (sim_grid >= threshold).astype(np.uint8)
    mask_px = np.array(
        Image.fromarray(mask_patch * 255).resize(orig.size, Image.NEAREST)
    ) > 0

    orig_arr = np.array(orig.convert("RGB"), dtype=np.float32)
    result = np.where(
        mask_px[..., None],
        colored * overlay_alpha + orig_arr * (1.0 - overlay_alpha),
        orig_arr * 0.25,
    ).clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(result_img)
    pw, ph = orig.width / gw, orig.height / gh
    r = max(5, int(min(pw, ph) * 0.35))
    for ci, cj in clicked_list:
        cx = int((cj + 0.5) * pw)
        cy = int((ci + 0.5) * ph)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=2)
        draw.ellipse([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2], outline=(0, 0, 0), width=1)

    n_above = int(mask_patch.sum())
    info = {
        "n_points": len(clicked_list),
        "threshold": round(float(threshold), 2),
        "above_threshold": n_above,
        "coverage": f"{100 * n_above / (gh * gw):.1f}%",
        "grid": f"{gh}x{gw}",
    }
    return result_img, info


def run_similarity(
    img: Image.Image,
    model_id: str,
    size: int,
    click_x: float,  # fraction of image width  [0, 1]
    click_y: float,  # fraction of image height [0, 1]
    threshold: float,
    colormap: str,
    overlay_alpha: float,
):
    proc, model = get_backbone(model_id)
    patch = model.config.patch_size
    w, h = img.size
    scale = size / max(w, h)
    nw = max(patch, round(w * scale / patch) * patch)
    nh = max(patch, round(h * scale / patch) * patch)
    img_in = img.resize((nw, nh), Image.BICUBIC)
    inputs = proc(
        images=img_in, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to(device())

    with torch.no_grad():
        out = model(**inputs)
    tokens = out.last_hidden_state[0]

    _, _, H, W = inputs["pixel_values"].shape
    gh, gw = H // patch, W // patch
    n_patches = gh * gw
    patch_tokens = tokens[-n_patches:].float().cpu()  # (n_patches, D)

    # Map fractional click to patch grid cell
    ci = int(np.clip(click_y * gh, 0, gh - 1))
    cj = int(np.clip(click_x * gw, 0, gw - 1))
    query = patch_tokens[ci * gw + cj]  # (D,)

    # Cosine similarity: normalise then dot product
    q_norm = query / (query.norm() + 1e-8)
    t_norm = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + 1e-8)
    sim = (t_norm @ q_norm).numpy()  # (n_patches,) in [-1, 1]
    sim_grid = sim.reshape(gh, gw).astype(np.float32)

    token = _cache_sim(sim_grid, (ci, cj), (gh, gw), img)
    result_img, info = render_similarity(sim_grid, (ci, cj), img, threshold, colormap, overlay_alpha)
    return result_img, info, token


def render_similarity(
    sim_grid: np.ndarray,  # (gh, gw) cosine sim in [-1, 1]
    clicked: tuple,        # (ci, cj) patch grid coords
    orig: Image.Image,
    threshold: float,      # cosine sim threshold in [-1, 1]
    colormap: str,
    overlay_alpha: float,
) -> tuple[Image.Image, dict]:
    gh, gw = sim_grid.shape
    ci, cj = clicked

    # Normalise full range to [0, 1] for colormap (preserves relative contrast)
    lo, hi = sim_grid.min(), sim_grid.max()
    sim_01 = (sim_grid - lo) / (hi - lo + 1e-8)
    colored = np.array(apply_colormap(sim_01, colormap).resize(orig.size, Image.BILINEAR), dtype=np.float32)

    # Build threshold mask at patch resolution, resize to pixel space
    mask_patch = (sim_grid >= threshold).astype(np.uint8)
    mask_px = np.array(
        Image.fromarray(mask_patch * 255).resize(orig.size, Image.NEAREST)
    ) > 0  # (H, W) bool

    orig_arr = np.array(orig.convert("RGB"), dtype=np.float32)
    # Above threshold: blend colormap + original. Below: dim original to 30%.
    result = np.where(
        mask_px[..., None],
        colored * overlay_alpha + orig_arr * (1.0 - overlay_alpha),
        orig_arr * 0.25,
    ).clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    # Draw a small white + black dot at the clicked patch
    from PIL import ImageDraw
    draw = ImageDraw.Draw(result_img)
    pw, ph = orig.width / gw, orig.height / gh
    cx = int((cj + 0.5) * pw)
    cy = int((ci + 0.5) * ph)
    r = max(5, int(min(pw, ph) * 0.35))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=2)
    draw.ellipse([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2], outline=(0, 0, 0), width=1)

    n_above = int(mask_patch.sum())
    info = {
        "clicked_patch": f"({ci},{cj})",
        "threshold": round(float(threshold), 2),
        "above_threshold": n_above,
        "coverage": f"{100 * n_above / (gh * gw):.1f}%",
        "grid": f"{gh}x{gw}",
    }
    return result_img, info


# ---- segmentation palettes ------------------------------------------------
# Each palette is a list of (R, G, B) colours. We cycle through them when
# there are more clusters than colours.
_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "colorful": [
        (99, 179, 237), (252, 129, 74), (104, 211, 145), (246, 224, 94),
        (183, 148, 246), (247, 104, 161), (79, 209, 197), (237, 137, 54),
        (66, 153, 225), (72, 187, 120), (237, 100, 166), (160, 174, 192),
        (246, 173, 85), (129, 230, 217), (154, 230, 180), (250, 240, 137),
        (213, 63, 140), (49, 151, 149), (214, 158, 46), (90, 103, 216),
    ],
    "pastel": [
        (190, 227, 248), (254, 200, 154), (154, 230, 180), (250, 240, 137),
        (214, 188, 250), (251, 182, 206), (129, 230, 217), (246, 211, 101),
        (190, 212, 252), (154, 230, 180), (246, 173, 85), (203, 213, 224),
        (254, 215, 215), (198, 246, 213), (255, 245, 157), (179, 229, 252),
        (225, 190, 231), (178, 235, 242), (255, 224, 178), (200, 230, 201),
    ],
    "vivid": [
        (229, 62, 62), (49, 151, 149), (66, 153, 225), (214, 158, 46),
        (159, 122, 234), (246, 135, 179), (56, 178, 172), (237, 137, 54),
        (72, 187, 120), (99, 179, 237), (246, 224, 94), (160, 174, 192),
        (183, 148, 246), (104, 211, 145), (247, 104, 161), (79, 209, 197),
        (237, 100, 166), (129, 230, 217), (252, 129, 74), (213, 63, 140),
    ],
}


def _kmeans_numpy(X: np.ndarray, k: int, n_iter: int = 30) -> np.ndarray:
    """Pure-numpy k-means. Fast enough for patch grids (N~200-2000, D~384)."""
    rng = np.random.default_rng(42)
    centers = X[rng.choice(len(X), k, replace=False)]
    labels = np.zeros(len(X), dtype=np.int32)
    for _ in range(n_iter):
        # (N, k) squared distances
        diff = X[:, None, :] - centers[None, :, :]  # (N, k, D)
        dists = (diff * diff).sum(-1)
        new_labels = dists.argmin(axis=1).astype(np.int32)
        new_centers = np.array(
            [X[new_labels == i].mean(0) if (new_labels == i).any() else centers[i]
             for i in range(k)],
            dtype=np.float32,
        )
        if np.array_equal(new_labels, labels) and np.allclose(centers, new_centers):
            break
        labels, centers = new_labels, new_centers
    return labels


def run_segmentation(
    img: Image.Image,
    model_id: str,
    size: int,
    n_clusters: int,
    pct_low: float,
    pct_high: float,
    overlay_alpha: float,
    palette: str,
    show_boundaries: bool,
):
    proc, model = get_backbone(model_id)
    patch = model.config.patch_size
    w, h = img.size
    scale = size / max(w, h)
    nw = max(patch, round(w * scale / patch) * patch)
    nh = max(patch, round(h * scale / patch) * patch)
    img_in = img.resize((nw, nh), Image.BICUBIC)
    inputs = proc(
        images=img_in, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to(device())

    with torch.no_grad():
        out = model(**inputs)
    tokens = out.last_hidden_state[0]

    _, _, H, W = inputs["pixel_values"].shape
    gh, gw = H // patch, W // patch
    n_patches = gh * gw
    feats = tokens[-n_patches:].float().cpu().numpy()  # (n_patches, D)

    # Optionally whiten features (PCA-based) for better cluster separation.
    feats = feats - feats.mean(0, keepdims=True)
    std = feats.std(0, keepdims=True) + 1e-8
    feats = feats / std

    k = max(2, min(int(n_clusters), n_patches))
    labels = _kmeans_numpy(feats, k)  # (n_patches,)

    colors = _PALETTES.get(palette, _PALETTES["colorful"])
    seg_rgb = np.zeros((gh * gw, 3), dtype=np.uint8)
    for i in range(k):
        seg_rgb[labels == i] = colors[i % len(colors)]
    seg_img = Image.fromarray(seg_rgb.reshape(gh, gw, 3)).resize(img.size, Image.NEAREST)

    if show_boundaries:
        seg_arr = np.array(seg_img)
        lbl_grid = labels.reshape(gh, gw)
        # Mark pixels where adjacent patches differ
        bound_h = np.repeat(
            np.repeat((lbl_grid[:-1] != lbl_grid[1:]).astype(np.uint8), patch, axis=0),
            patch, axis=1,
        )
        bound_v = np.repeat(
            np.repeat((lbl_grid[:, :-1] != lbl_grid[:, 1:]).astype(np.uint8), patch, axis=0),
            patch, axis=1,
        )
        bh = np.array(Image.fromarray(bound_h * 255).resize(img.size, Image.NEAREST)) > 0
        bv = np.array(Image.fromarray(bound_v * 255).resize(img.size, Image.NEAREST)) > 0
        boundary = bh | bv
        seg_arr[boundary] = [255, 255, 255]
        seg_img = Image.fromarray(seg_arr)

    # Blend with original
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    if alpha < 1.0:
        orig_arr = np.array(img.convert("RGB"), dtype=np.float32)
        seg_arr = np.array(seg_img, dtype=np.float32)
        blended = (seg_arr * alpha + orig_arr * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        seg_img = Image.fromarray(blended)

    # Compute per-cluster area fractions
    total = gh * gw
    area_chips = {f"seg_{i}": f"{100 * (labels == i).sum() / total:.1f}%" for i in range(k)}

    info = {
        "grid": f"{gh}x{gw}",
        "patches": int(n_patches),
        "clusters": k,
        "dim": int(tokens.shape[-1]),
        "input_size": f"{nw}x{nh}",
        "patch_size": int(patch),
        **area_chips,
    }
    return seg_img, info


# ---- app ------------------------------------------------------------------
app = FastAPI(title="DINOv3 Playground")

# Inference is CPU/GPU-bound and must run off the event loop (via threadpool) or
# it freezes the whole UI. The lock then serializes those threaded runs so two
# concurrent generates can't contend for MPS / the shared model at once — the
# second waits without blocking the event loop (page + config stay responsive).
_INFER_LOCK = asyncio.Lock()


@app.on_event("shutdown")
def _stop_downloads():
    """Cancel any in-flight downloads so the server exits cleanly."""
    with _DL_LOCK:
        for job in _DL_JOBS.values():
            job.stop()
        _DL_JOBS.clear()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


# Cap on a proxied download so a hostile/huge URL can't exhaust memory.
PROXY_MAX_BYTES = 25 * 1024 * 1024  # 25 MB


def _fetch(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read(PROXY_MAX_BYTES + 1), resp.headers.get("Content-Type", "")


def _is_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def _og_image(html_bytes: bytes, base_url: str) -> str | None:
    """If a page (not an image) was fetched, dig out its og:image / first <img>.
    Lets you drop a *link to a page* (or a search-result thumbnail) and still get
    the picture it points at."""
    html = html_bytes[:300_000].decode("utf-8", "ignore")
    pats = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
        r'<img[^>]+src=["\'](https?:[^"\']+)["\']',
    ]
    for p in pats:
        m = re.search(p, html, re.I)
        if m:
            return urllib.parse.urljoin(base_url, m.group(1))
    return None


@app.get("/api/proxy")
def proxy(url: str):
    """Fetch a remote image server-side so the browser can drop an image dragged
    from another web page/tab without tripping over CORS. Returns the raw bytes.

    If the URL turns out to be an HTML page rather than an image (e.g. a dropped
    page link, or a search-result wrapper), we fall back to its og:image once.
    """
    if not url.lower().startswith(("http://", "https://")):
        return JSONResponse({"error": "only http/https URLs are allowed"}, status_code=400)
    try:
        data, ctype = _fetch(url)
        if not _is_image(data) and not ctype.startswith("image/"):
            follow = _og_image(data, url)  # page, not image -> try its og:image
            if follow:
                data, ctype = _fetch(follow)
    except Exception as e:  # network/HTTP errors -> surface to the UI
        return JSONResponse({"error": f"could not fetch image: {e}"}, status_code=502)
    if len(data) > PROXY_MAX_BYTES:
        return JSONResponse({"error": "image too large (>25 MB)"}, status_code=413)
    if not _is_image(data):
        return JSONResponse(
            {"error": "the URL did not point to an image (try dragging the image itself, "
                      "or open it in its own tab and drag from there)"},
            status_code=415,
        )
    media = ctype if ctype.startswith("image/") else "image/png"
    return Response(content=data, media_type=media)


def _depth_model_list():
    return [
        {
            "key": k,
            "label": v["label"],
            "repo": v["repo"],
            "desc": v["desc"],
            "gated": v.get("gated", False),
            "metric": v.get("metric", False),
            "size_mb": v.get("size_mb"),
            "backbone": v.get("backbone", "—"),
            "dataset": v.get("dataset", "—"),
            "loader": v.get("loader", "pipeline"),
            "cached": _is_cached(v["repo"]),
        }
        for k, v in DEPTH_MODELS.items()
    ]


@app.get("/api/config")
def config():
    return {
        "pca_models": list(PCA_MODELS.keys()),
        "default_pca_model": "vits16",
        "depth_models": _depth_model_list(),
        "default_depth_model": DEFAULT_DEPTH_KEY,
        "live_depth_model": LIVE_DEPTH_KEY,
        "device": device(),
        "colormaps": ["turbo", "magma", "gray"],
        "seg_palettes": list(_PALETTES.keys()),
    }


@app.get("/api/depth/models")
def depth_models():
    """Fresh cached-status for the depth dropdown (re-poll after a download)."""
    return {"models": _depth_model_list(), "default": DEFAULT_DEPTH_KEY}


@app.get("/api/depth/download")
async def download_depth(model: str, request: Request):
    """Stream HF download progress for a depth model as Server-Sent Events.

    Emits `data: {downloaded,total,pct,done,error}` lines until the snapshot
    download finishes (or errors). If the model is already cached it returns a
    single done event immediately. If the client disconnects (tab closed /
    page refreshed) the stream ends and — if it was the last viewer — the
    download is cancelled; HF resumes it from the partial file next time.
    """
    spec = DEPTH_MODELS.get(model)
    if spec is None:
        return JSONResponse({"error": f"unknown model '{model}'"}, status_code=404)
    repo = spec["repo"]

    async def stream():
        if _is_cached(repo):
            yield f"data: {json.dumps({'pct': 100, 'done': True, 'cached': True})}\n\n"
            return
        # Starting a job touches the network/spawns a process — keep it off the
        # event loop.
        job = await run_in_threadpool(_acquire_job, repo)
        try:
            while True:
                if await request.is_disconnected():
                    break  # finally-block releases (and maybe cancels) the job
                done = job.done
                bytes_done = (
                    job.total if (done and job.success)
                    else await run_in_threadpool(_downloaded_bytes, repo)
                )
                pct = round(min(bytes_done / job.total * 100, 100), 1) if job.total else 0.0
                payload = {
                    "downloaded": bytes_done,
                    "total": job.total,
                    "pct": pct,
                    "done": done,
                    "error": job.error,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                if done:
                    break
                await asyncio.sleep(0.5)
        finally:
            await run_in_threadpool(_release_job, repo)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/pca")
async def api_pca(
    image: UploadFile = File(...),
    model: str = Form("vits16"),
    size: int = Form(448),
    pct_low: float = Form(2.0),
    pct_high: float = Form(98.0),
    n_components: int = Form(3),
    remove_bg: bool = Form(False),
    bg_threshold: float = Form(0.5),
    side_by_side_out: bool = Form(False),
    max_side: int = Form(1024),
):
    model_id = PCA_MODELS.get(model, PCA_MODELS["vits16"])
    img = read_image(await image.read(), max_side)
    t0 = time.time()
    try:
        async with _INFER_LOCK:
            result, info = await run_in_threadpool(
                run_pca, img, model_id, size, pct_low, pct_high,
                n_components, remove_bg, bg_threshold,
            )
    except Exception as e:  # surface model/auth errors to the UI
        return JSONResponse({"error": str(e)}, status_code=500)
    out = side_by_side(img, result) if side_by_side_out else result
    info["model"] = model_id
    info["working_res"] = f"{img.width}x{img.height}"
    info["seconds"] = round(time.time() - t0, 2)
    return {"image": to_b64_png(out), "info": info}


@app.post("/api/depth")
async def api_depth(
    image: UploadFile = File(...),
    model: str = Form(DEFAULT_DEPTH_KEY),
    colormap: str = Form("turbo"),
    pct_low: float = Form(2.0),
    pct_high: float = Form(98.0),
    near: float = Form(0.0),
    far: float = Form(0.0),
    gamma: float = Form(1.0),
    side_by_side_out: bool = Form(False),
    max_side: int = Form(1024),
):
    # Don't trigger a slow blocking download on a generate request — make the
    # client download the model first (the UI streams progress for that).
    spec = DEPTH_MODELS.get(model, DEPTH_MODELS[DEFAULT_DEPTH_KEY])
    if not _is_cached(spec["repo"]):
        return JSONResponse(
            {"error": f"model '{spec['label']}' is not downloaded yet — "
                      "download it first, then generate.", "needs_download": True},
            status_code=409,
        )
    img = read_image(await image.read(), max_side)
    t0 = time.time()
    try:
        async with _INFER_LOCK:
            result, info, depth, metric = await run_in_threadpool(
                run_depth, img, colormap, model, pct_low, pct_high, near, far, gamma
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    token = _cache_depth(depth, metric, spec["repo"], img)
    out = side_by_side(img, result) if side_by_side_out else result
    info["working_res"] = f"{img.width}x{img.height}"
    info["seconds"] = round(time.time() - t0, 2)
    return {"image": to_b64_png(out), "info": info, "token": token}


@app.post("/api/depth/recolor")
async def api_depth_recolor(
    token: str = Form(...),
    colormap: str = Form("turbo"),
    pct_low: float = Form(2.0),
    pct_high: float = Form(98.0),
    near: float = Form(0.0),
    far: float = Form(0.0),
    gamma: float = Form(1.0),
    side_by_side_out: bool = Form(False),
):
    """Re-colour a previously computed depth map (no inference) so display
    controls update live. 410 if the cached depth has been evicted -> the client
    should re-generate."""
    with _DEPTH_CACHE_LOCK:
        entry = _DEPTH_CACHE.get(token)
        if entry is not None:
            _DEPTH_CACHE.move_to_end(token)  # keep hot entries from being evicted
    if entry is None:
        return JSONResponse({"error": "depth expired — regenerate", "stale": True}, status_code=410)
    orig = entry["orig"]
    depth_img, info = await run_in_threadpool(
        render_depth, entry["depth"], entry["metric"], entry["repo"], orig.size,
        colormap, pct_low, pct_high, near, far, gamma,
    )
    out = side_by_side(orig, depth_img) if side_by_side_out else depth_img
    info["working_res"] = f"{orig.width}x{orig.height}"
    return {"image": to_b64_png(out), "info": info}


@app.post("/api/depth_frame")
async def api_depth_frame(
    image: UploadFile = File(...),
    model: str = Form(LIVE_DEPTH_KEY),
    colormap: str = Form("turbo"),
    gamma: float = Form(1.0),
    near: float = Form(0.0),
    far: float = Form(0.0),
    pct_low: float = Form(2.0),
    pct_high: float = Form(98.0),
    max_side: int = Form(320),
):
    """One frame of the live-camera depth stream. Small max_side for speed,
    serialized through _INFER_LOCK so overlapping frames don't fight over the
    device. No depth caching (frames are transient)."""
    spec = DEPTH_MODELS.get(model, DEPTH_MODELS[LIVE_DEPTH_KEY])
    if not _is_cached(spec["repo"]):
        return JSONResponse(
            {"error": "live model not downloaded", "needs_download": True,
             "model": model}, status_code=409,
        )
    img = read_image(await image.read(), max_side)
    t0 = time.time()
    try:
        async with _INFER_LOCK:
            depth_img, info, _, _ = await run_in_threadpool(
                run_depth, img, colormap, model, pct_low, pct_high, near, far, gamma
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    info["ms"] = round((time.time() - t0) * 1000)
    return {"image": to_b64_png(depth_img), "info": info}


@app.post("/api/depth_pca")
async def api_depth_pca(
    image: UploadFile = File(...),
    model: str = Form(DEFAULT_DEPTH_KEY),
    size: int = Form(448),
    pct_low: float = Form(2.0),
    pct_high: float = Form(98.0),
    side_by_side_out: bool = Form(False),
    max_side: int = Form(1024),
):
    spec = DEPTH_MODELS.get(model, DEPTH_MODELS[DEFAULT_DEPTH_KEY])
    if spec.get("loader") == "da3":
        return JSONResponse(
            {"error": "Feature PCA isn't available for Depth Anything 3 models "
                      "(they don't expose a transformers backbone). Use the Depth "
                      "map view, or pick a non-DA3 model."},
            status_code=400,
        )
    if not _is_cached(spec["repo"]):
        return JSONResponse(
            {"error": f"model '{spec['label']}' is not downloaded yet — "
                      "download it first, then generate.", "needs_download": True},
            status_code=409,
        )
    img = read_image(await image.read(), max_side)
    t0 = time.time()
    try:
        async with _INFER_LOCK:
            result, info = await run_in_threadpool(
                run_depth_pca, img, model, size, pct_low, pct_high
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    out = side_by_side(img, result) if side_by_side_out else result
    info["working_res"] = f"{img.width}x{img.height}"
    info["seconds"] = round(time.time() - t0, 2)
    return {"image": to_b64_png(out), "info": info}


@app.post("/api/segment")
async def api_segment(
    image: UploadFile = File(...),
    model: str = Form("vits16"),
    size: int = Form(448),
    n_clusters: int = Form(8),
    pct_low: float = Form(2.0),
    pct_high: float = Form(98.0),
    overlay_alpha: float = Form(0.7),
    palette: str = Form("colorful"),
    show_boundaries: bool = Form(True),
    side_by_side_out: bool = Form(False),
    max_side: int = Form(1024),
):
    model_id = PCA_MODELS.get(model, PCA_MODELS["vits16"])
    img = read_image(await image.read(), max_side)
    t0 = time.time()
    try:
        async with _INFER_LOCK:
            result, info = await run_in_threadpool(
                run_segmentation, img, model_id, size, n_clusters,
                pct_low, pct_high, overlay_alpha, palette, show_boundaries,
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    out = side_by_side(img, result) if side_by_side_out else result
    info["model"] = model_id
    info["working_res"] = f"{img.width}x{img.height}"
    info["seconds"] = round(time.time() - t0, 2)
    return {"image": to_b64_png(out), "info": info}


@app.post("/api/similarity")
async def api_similarity(
    image: UploadFile = File(...),
    model: str = Form("vits16"),
    size: int = Form(448),
    click_x: float = Form(0.5),
    click_y: float = Form(0.5),
    threshold: float = Form(0.5),
    colormap: str = Form("turbo"),
    overlay_alpha: float = Form(0.75),
    max_side: int = Form(1024),
):
    model_id = PCA_MODELS.get(model, PCA_MODELS["vits16"])
    img = read_image(await image.read(), max_side)
    t0 = time.time()
    try:
        async with _INFER_LOCK:
            result, info, token = await run_in_threadpool(
                run_similarity, img, model_id, size, click_x, click_y,
                threshold, colormap, overlay_alpha,
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    info["model"] = model_id
    info["working_res"] = f"{img.width}x{img.height}"
    info["seconds"] = round(time.time() - t0, 2)
    return {"image": to_b64_png(result), "info": info, "token": token}


@app.post("/api/similarity/rerender")
async def api_similarity_rerender(
    token: str = Form(...),
    threshold: float = Form(0.5),
    colormap: str = Form("turbo"),
    overlay_alpha: float = Form(0.75),
):
    """Re-render cached similarity map with new threshold/colormap — no model re-run."""
    with _SIM_CACHE_LOCK:
        entry = _SIM_CACHE.get(token)
        if entry is not None:
            _SIM_CACHE.move_to_end(token)
    if entry is None:
        return JSONResponse({"error": "similarity expired — click again", "stale": True}, status_code=410)
    # Multi-point entries store all clicked points; single entries use one.
    clicked_all = entry.get("clicked_all")
    if clicked_all:
        result_img, info = await run_in_threadpool(
            render_similarity_multi, entry["sim_map"], clicked_all, entry["orig"],
            threshold, colormap, overlay_alpha,
        )
    else:
        result_img, info = await run_in_threadpool(
            render_similarity, entry["sim_map"], entry["clicked"], entry["orig"],
            threshold, colormap, overlay_alpha,
        )
    return {"image": to_b64_png(result_img), "info": info}


@app.post("/api/similarity/multi")
async def api_similarity_multi(
    image: UploadFile = File(...),
    model: str = Form("vits16"),
    size: int = Form(448),
    points: str = Form("[]"),   # JSON array of [cx, cy] pairs
    threshold: float = Form(0.5),
    colormap: str = Form("turbo"),
    overlay_alpha: float = Form(0.75),
    max_side: int = Form(1024),
):
    model_id = PCA_MODELS.get(model, PCA_MODELS["vits16"])
    pts = json.loads(points)
    if not pts:
        return JSONResponse({"error": "no points provided"}, status_code=400)
    img = read_image(await image.read(), max_side)
    t0 = time.time()
    try:
        async with _INFER_LOCK:
            result, info, token = await run_in_threadpool(
                run_similarity_multi, img, model_id, size, pts,
                threshold, colormap, overlay_alpha,
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    info["model"] = model_id
    info["working_res"] = f"{img.width}x{img.height}"
    info["seconds"] = round(time.time() - t0, 2)
    return {"image": to_b64_png(result), "info": info, "token": token}
