# Train the `dino_dt` policy on an H100 (Brev)

End-to-end runbook: provision an H100 on [Brev](https://brev.dev), train the
DINOv3 diffusion-transformer policy (`dino_dt`) on
`hdhruva/makermods_pick_book_place`, push the result to the Hub, then tear the
instance down.

> Cost warning: an H100 bills by the minute. **Always run the teardown step
> (Section 8)** when finished — a forgotten instance is the expensive mistake.

---

## 0. Prerequisites (do once, on your laptop)

- A Brev account: <https://console.brev.dev>
- A Hugging Face account, and **accept the DINOv3 license** so the gated weights
  download — open <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>
  and click "Agree". Without this, training fails with a 401 on the backbone.
- (Optional) A Weights & Biases account for live loss curves.

---

## 1. Install the Brev CLI and log in (laptop)

```bash
# macOS (Homebrew)
brew install brevdev/homebrew-brev/brev

# …or any OS
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh | sh

brev login        # opens a browser to authenticate
```

---

## 2. Provision an H100

### First, see what's actually available (prices/stock change)

```bash
brev search gpu --gpu-name H100 --sort price
```

Snapshot of the **single-H100** options at time of writing (sorted by price; 1× H100 80 GB is plenty for `dino_dt`):

| Instance type (`--type`)                | Provider     | GPUs | VRAM  | Disk     | Stoppable | $/hr  |
|-----------------------------------------|--------------|------|-------|----------|-----------|-------|
| `hyperstack_H100`                       | hyperstack   | 1    | 80 GB | 850 GB   | no        | **$2.28** |
| `scaleway_H100`                         | scaleway     | 1    | 80 GB | 1 TB     | no        | $3.96 |
| `gpu-h100-sxm.1gpu-16vcpu-200gb`        | nebius       | 1    | 80 GB | 50GB–3TB | **yes (S)** | $4.62 |
| `gpu_1x_h100_sxm5`                      | lambda-labs  | 1    | 80 GB | 3 TB     | reboot (R) | $5.15 |
| `paperspace_H100`                       | paperspace   | 1    | 80 GB | 500 GB   | no        | $7.19 |

(There are also 2×/4×/8× H100 boxes — `hyperstack_H100x2` $4.56, `…x8` $18.24, etc. — but a single GPU is all `dino_dt` needs, so skip them.)

- **Cheapest:** `hyperstack_H100` ($2.28/hr). Fine for a one-shot run: train → push to Hub → delete.
- **Want stop/resume** (pause billing without losing the disk): pick the nebius
  `gpu-h100-sxm.1gpu-16vcpu-200gb` ($4.62/hr, the only **Stoppable** single-H100 here).

### Create it (with automatic fallback, cheapest first)

```bash
# Tries each single-H100 type in order until one provisions
brev create dino-h100 \
  --type hyperstack_H100,scaleway_H100,gpu-h100-sxm.1gpu-16vcpu-200gb,gpu_1x_h100_sxm5,paperspace_H100

brev ls          # wait until dino-h100 shows RUNNING
```

Shortcuts:
```bash
brev create dino-h100 --type hyperstack_H100   # pin the cheapest one specifically
brev create dino-h100 -g h100 --dry-run        # preview what it would pick, no charge
```

---

## 3. Connect to the instance

```bash
brev shell dino-h100         # opens an SSH shell on the H100
# (or: brev open dino-h100   to open it in VS Code)
```

Everything below runs **on the H100**.

---

## 4. Set up the environment (on the H100)

```bash
# Sanity check the GPU is visible
nvidia-smi                   # should list an H100

# Install Miniforge if conda isn't already on the box (x86_64 Linux).
# Miniforge defaults to the conda-forge channel, which avoids the Anaconda
# Terms-of-Service prompt that Miniconda's default channels now trigger.
if ! command -v conda &>/dev/null; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda init bash              # so 'conda activate' works in new shells
fi

# If you already installed Miniconda and hit the ToS error, either reinstall with
# Miniforge above, or accept the ToS once:
#   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
#   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Clone the repo (lerobot-MakerMods lives inside it)
git clone https://github.com/Msrikrishna/robot-manipulation-lerobot.git
cd robot-manipulation-lerobot/lerobot-MakerMods

# Python env (Linux torch wheels include CUDA by default)
conda create -n makermods python=3.10 -y
conda activate makermods

# FFmpeg is required by torchcodec to decode the dataset's episode videos (mp4).
# A bare GPU box has none -> "Could not load libtorchcodec".
# IMPORTANT: pin to <8. FFmpeg 8 ships libavutil.so.60, which this torchcodec
# range does NOT support (it looks for .so.56-.59 = FFmpeg 4-7) -> still fails.
conda install -y -c conda-forge "ffmpeg<8"

pip install -e .   # fork + ALL python deps (now includes transformers + feetech-servo-sdk)

# Verify CUDA torch + transformers + torchcodec all import
python -c "import torch, transformers, torchcodec; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0), '| transformers', transformers.__version__, '| torchcodec', torchcodec.__version__)"
```

---

## 5. Authenticate (on the H100)

```bash
huggingface-cli login        # paste an HF token (read access to dataset + gated DINOv3)
wandb login                  # optional; or set --wandb.enable=false in the next step
```

---

## 6. Train `dino_dt` (on the H100)

Run inside `tmux` so it survives an SSH disconnect:

```bash
tmux new -s train

conda activate makermods     # re-activate inside the new tmux pane
cd ~/lerobot-MakerMods

lerobot-train \
  --dataset.repo_id=hdhruva/makermods_pick_book_place \
  --policy.type=dino_dt \
  --policy.device=cuda \
  --output_dir=outputs/train/dino_dt_pick_book_place \
  --job_name=dino_dt_pick_book_place \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=10000 \
  --num_workers=8 \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=srik410/dino_dt_pick_book_place   # YOUR HF account (not the dataset owner's)
```

Notes:
- `dino_dt` uses a **frozen DINOv3 ViT-S/16** backbone (`facebook/dinov3-vits16-pretrain-lvd1689m`,
  downloaded on first run) + a diffusion denoiser trained from scratch.
- `--batch_size=32` is comfortable on an 80 GB H100; raise toward 64 if memory allows.
- Detach from tmux with `Ctrl-b d`; reattach later with `tmux attach -t train`.
- Checkpoints land in `outputs/train/dino_dt_pick_book_place/checkpoints/`.
  Resume with `--resume=true` and the same `--output_dir`.
- The final policy is pushed to `srik410/dino_dt_pick_book_place` on the Hub
  (your account — the `hdhruva` namespace is the dataset owner's, you can't push there).

---

## 7. Monitor

- **wandb**: open the run URL printed at startup.
- **GPU**: in another pane/shell on the instance: `watch -n2 nvidia-smi`.
- **Logs**: training prints loss every `--log_freq` (default 200) steps.

---

## 8. Tear down (IMPORTANT — stops billing)

From your **laptop**:

```bash
brev stop dino-h100      # stops compute; keeps the disk (small storage cost)
# …or, to delete everything and stop all charges:
brev delete dino-h100

brev ls                  # confirm it's stopped/gone
```

> Make sure your trained policy finished pushing to the Hub (Section 6) before
> you `delete` — deleting wipes the disk and any local checkpoints.

---

## Quick reference

| Step | Where | Command |
|------|-------|---------|
| Login | laptop | `brev login` |
| Connect | laptop | `brev shell dino-h100` |
| Install | H100 | `pip install -e .` |
| Auth | H100 | `huggingface-cli login` |
| Train | H100 | `lerobot-train --policy.type=dino_dt …` |
| Teardown | laptop | `brev delete dino-h100` |

> Note: the cheapest box (`hyperstack_H100`) is **not stoppable** — `brev stop` won't
> pause billing, only `brev delete` does. Pick the Nebius single-H100 type if you need
> stop/resume.

---

## Troubleshooting — issues we actually hit

### 1. `CondaToSNonInteractiveError` (Terms of Service not accepted)
Miniconda's default channels (`pkgs/main`, `pkgs/r`) now require ToS acceptance.
**Fix:** use **Miniforge** (Section 4 already does — it defaults to conda-forge, no ToS).
If you already have Miniconda: `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (and `.../pkgs/r`).

### 2. `RuntimeError: Could not load libtorchcodec` (the big one)
torchcodec decodes the dataset's mp4 episodes and needs FFmpeg's **shared libs**
(`libavutil.so.*`) — which pip does NOT provide. Two distinct failure modes:
- **No FFmpeg at all** → `libavutil.so.59/58/57/56: cannot open shared object file`.
- **FFmpeg 8 installed** → env has `libavutil.so.60`, which this torchcodec doesn't
  support (it only knows FFmpeg 4-7).

**Diagnose:**
```bash
ls "$CONDA_PREFIX"/lib/libavutil.so*    # .so.60 = FFmpeg 8 (too new); none = not installed
```
**Fix:** install FFmpeg **< 8** into the active env:
```bash
conda install -y -c conda-forge "ffmpeg<8"
python -c "import torchcodec; print('ok', torchcodec.__version__)"
```
(`pip install ffmpeg/imageio-ffmpeg` does NOT fix this — those ship a binary, not the
linkable `.so` libs.)

### 3. `ModuleNotFoundError: transformers` / `scservo_sdk`
These are now declared in `pyproject.toml`, so `pip install -e .` covers them. If you're
on an OLD clone that predates that change: `pip install "transformers>=4.53.0" feetech-servo-sdk`.

### 4. `FileExistsError: Output directory … already exists and resume is False`
A previous (often failed) run left the output dir. Either delete it or use a new name:
```bash
rm -rf outputs/train/<job_name>          # if it has no checkpoint worth keeping
# or change --output_dir / --job_name to a new value
```
(Only use `--resume=true` if `checkpoints/` actually has a real checkpoint.)

### 5. 401 / gated weights on `facebook/dinov3-...`
The DINOv3 backbone is gated. Accept the license at
<https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m> and `huggingface-cli login`.

### 6. Stopping early + pushing the checkpoint yourself
`Ctrl-C` stops training but **skips the auto-push** (that only runs at the very end).
The checkpoint is still on disk — upload it to **your** account:
```bash
hf upload srik410/<name> outputs/train/<job_name>/checkpoints/last/pretrained_model
```
Push to `srik410/...` (your account) — NOT `hdhruva/...` (the dataset owner's; no write access).

### 7. Verify a pushed policy actually loads
```bash
conda activate makermods
python -c "from lerobot.policies.dino_dt.modeling_dino_dt import DinoDTPolicy; \
  p=DinoDTPolicy.from_pretrained('srik410/<name>'); \
  print('ok', sum(x.numel() for x in p.parameters())/1e6, 'M params')"
```
