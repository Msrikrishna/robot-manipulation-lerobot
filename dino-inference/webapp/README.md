# DINOv3 Playground (web UI)

A local web app to drag-and-drop an image, tweak parameters, and generate either
a **PCA feature map** or a **depth map** with the DINOv3 models in this repo.

## Run

```bash
# from the dino-inference/ root
uv run uvicorn webapp.app:app --reload --port 8000
# open http://localhost:8000
```

Or use the helper:

```bash
./webapp/run.sh
```

First generation downloads/loads the model (cached after that). Weights are gated —
accept the license on the model page and run `huggingface-cli login` once.

## Modes & controls

**PCA features** — projects patch tokens to 3 PCA components → RGB.
- **Backbone**: vits16 (fast) … vitl16 (richer, slower)
- **Input size**: longest side fed to the model; drives the patch grid (more patches = finer map)
- **Robust clip low/high**: per-channel percentile contrast stretch
- **Remove background**: blacks out patches below a threshold on PCA component 1

**Depth** — `dinov3-vitl16-chmv2-dpt-head` (DPT head). Pick a colormap (turbo/magma/gray).

Both modes have a **side-by-side with original** toggle and a **Download PNG** button.
The stats row reports patch grid, token dim, input size, and timing.

## API (if you want to script it)

- `GET /api/config` — models, device, colormaps
- `POST /api/pca` — multipart `image` + `model, size, pct_low, pct_high, remove_bg, bg_threshold, side_by_side_out`
- `POST /api/depth` — multipart `image` + `colormap, side_by_side_out`

Returns `{ "image": "data:image/png;base64,…", "info": {…} }`.
