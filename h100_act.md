# Train ACT on an H100 (Brev) — merged pick-book dataset

End-to-end runbook: provision an H100 on [Brev](https://brev.dev), train the stock
**ACT** policy (ResNet-18 backbone, CVAE action-chunking — fast and reliable) on
`srik410/makermods_pick_book_parallel_grasp_merged` (45 episodes), push the trained
policy to the Hub, then tear the instance down.

> Cost warning: an H100 bills by the minute. **Always run the teardown (Section 8)**
> when finished — the cheapest box (`hyperstack_H100`) is **not stoppable**, so only
> `brev delete` stops billing.

---

## 0. Prerequisites (on your laptop)

- A Brev account: <https://console.brev.dev>
- A Hugging Face account (`srik410`). The merged dataset is **public**, so no login is
  needed to *read* it — but you do need `hf auth login` to *push* the trained model.
- (Optional) A Weights & Biases account for live loss curves.

ACT uses a torchvision **ResNet-18** backbone (downloaded automatically) — there is **no
gated DINOv3** here, so no model license to accept (unlike the dino policies).

---

## 1. Install the Brev CLI and log in (laptop)

```bash
brew install brevdev/homebrew-brev/brev      # macOS
# or any OS: curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh | sh
brev login
```

---

## 2. Provision an H100

Check current availability/pricing first (it changes):
```bash
brev search gpu --gpu-name H100 --sort price
```

Single-H100 (1x 80 GB is plenty for ACT). Create with an automatic fallback chain,
cheapest first:
```bash
brev create act-h100 \
  --type hyperstack_H100,scaleway_H100,gpu-h100-sxm.1gpu-16vcpu-200gb,gpu_1x_h100_sxm5,paperspace_H100

brev ls          # wait until act-h100 shows RUNNING
```
(`hyperstack_H100` ~$2.28/hr is cheapest but not stoppable; the Nebius `gpu-h100-sxm.1gpu-...`
~$4.62/hr is the only stoppable single-H100 if you want pause/resume.)

---

## 3. Connect

```bash
brev shell act-h100          # SSH shell on the H100
```
Everything below runs **on the H100**.

---

## 4. Set up the environment (on the H100)

```bash
nvidia-smi                   # confirm an H100 is visible

# Install Miniforge if conda isn't present (conda-forge default -> no Anaconda ToS prompt)
if ! command -v conda &>/dev/null; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda init bash
fi

# Clone the repo (lerobot-MakerMods lives inside it)
git clone https://github.com/Msrikrishna/robot-manipulation-lerobot.git
cd robot-manipulation-lerobot/lerobot-MakerMods

# Python env (Linux torch wheels include CUDA by default)
conda create -n makermods python=3.10 -y
conda activate makermods

# FFmpeg is required by torchcodec to decode the dataset's episode videos (mp4).
# IMPORTANT: pin <8. FFmpeg 8 (libavutil.so.60) is NOT supported by this torchcodec
# range and causes "Could not load libtorchcodec".
conda install -y -c conda-forge "ffmpeg<8"

pip install -e .   # fork + deps

# Safety net: lerobot-train imports the dino_dt policy at load time, which needs
# transformers. It's declared in pyproject, but install explicitly in case the
# cloned commit predates that:
pip install "transformers>=4.53.0"

# Verify CUDA + torchcodec load
python -c "import torch, torchcodec; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0), '| torchcodec', torchcodec.__version__)"
```

---

## 5. Authenticate (on the H100)

```bash
hf auth login                # paste an HF token with WRITE access (to push the model)
wandb login                  # optional; or set --wandb.enable=false below
```

---

## 6. Train ACT (on the H100)

Run inside `tmux` so it survives an SSH disconnect:

```bash
tmux new -s train

conda activate makermods
cd ~/robot-manipulation-lerobot/lerobot-MakerMods

lerobot-train \
  --dataset.repo_id=srik410/makermods_pick_book_parallel_grasp_merged \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_pick_book_parallel_grasp \
  --job_name=act_pick_book_parallel_grasp \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=10000 \
  --num_workers=16 \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=srik410/act_pick_book_parallel_grasp
```

Notes:
- **ACT = ResNet-18 (ImageNet, trainable) + CVAE transformer.** Single forward pass, so
  it trains and infers fast; `--batch_size=64` is comfortable on an 80 GB H100.
- Loss is **`l1_loss + kl_loss`** — watch the **`l1_loss`** (the total looks large because
  the KL term is weighted ×10). ACT typically plateaus ~50–100K steps; with this dataset
  size it may flatten earlier.
- Detach from tmux: `Ctrl-b d`; reattach: `tmux attach -t train`.
- Checkpoints in `outputs/train/act_pick_book_parallel_grasp/checkpoints/`; resume with
  `--resume=true` and the same `--output_dir`.
- The final policy is pushed to `srik410/act_pick_book_parallel_grasp` (your account).
- Want DINOv3 vision instead of ResNet? swap `--policy.type=act_dino` (needs `hf auth login`
  + the gated DINOv3 license) — see `brev-H100-dino.md`.

---

## 7. Monitor

- **wandb**: open the run URL printed at startup (project `lerobot`).
- **GPU**: in another shell on the box — `watch -n2 nvidia-smi`.
- **Logs**: a `step / l1_loss / grad_norm` line every `--log_freq` (200) steps.

---

## 8. Tear down (IMPORTANT — stops billing)

From your **laptop**:
```bash
brev delete act-h100     # hyperstack is not stoppable; delete stops all charges
brev ls                  # confirm it's gone
```
> Make sure the policy finished pushing (Section 6) — or upload the checkpoint manually —
> before you delete, since deleting wipes the disk:
> `hf upload srik410/act_pick_book_parallel_grasp outputs/train/act_pick_book_parallel_grasp/checkpoints/last/pretrained_model`

---

## Troubleshooting (same gotchas as the dino runbook)

- **`CondaToSNonInteractiveError`** → use Miniforge (Section 4), or
  `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (+ `/pkgs/r`).
- **`Could not load libtorchcodec` / `libavutil.so.* not found`** → install FFmpeg **<8**:
  `conda install -y -c conda-forge "ffmpeg<8"` (check `ls "$CONDA_PREFIX"/lib/libavutil.so*` —
  `.so.60` means FFmpeg 8, too new).
- **`ModuleNotFoundError: transformers`** → `pip install "transformers>=4.53.0"` (a `git pull`
  never re-installs deps; re-run `pip install -e .` or install it directly).
- **`FileExistsError: Output directory ... already exists`** → `rm -rf outputs/train/act_pick_book_parallel_grasp`
  or use a new `--output_dir`/`--job_name`.

---

## Quick reference

| Step | Where | Command |
|------|-------|---------|
| Login | laptop | `brev login` |
| Provision | laptop | `brev create act-h100 --type hyperstack_H100,...` |
| Connect | laptop | `brev shell act-h100` |
| Install | H100 | `conda install -c conda-forge "ffmpeg<8"` then `pip install -e .` |
| Auth | H100 | `hf auth login` |
| Train | H100 | `lerobot-train --policy.type=act --dataset.repo_id=srik410/makermods_pick_book_parallel_grasp_merged ...` |
| Teardown | laptop | `brev delete act-h100` |
