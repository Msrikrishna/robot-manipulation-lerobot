# MakerMods App — Setup & Run Cheatsheet

How to bring the whole app (FastAPI backend + Next.js frontend) up from scratch,
assuming `MakerMods-App/` and `lerobot-MakerMods/` are already cloned and **no env exists yet**.

Requires `conda` and `node`/`npm` already installed on the machine.

---

## 1. One-time setup

### Python env + robot engine

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install -e ~/Desktop/makermods-hackathon/lerobot-MakerMods
pip install -r ~/Desktop/makermods-hackathon/MakerMods-App/requirements.txt
```

### Frontend deps

```bash
cd ~/Desktop/makermods-hackathon/MakerMods-App/frontend
npm install
```

---

## 2. Run it (every time) — two terminals

```bash
# Terminal A — backend (:8000)
conda activate lerobot
cd ~/Desktop/makermods-hackathon/MakerMods-App
python -m backend.main
```

```bash
# Terminal B — frontend (:3000)
cd ~/Desktop/makermods-hackathon/MakerMods-App/frontend
npm run dev
```

Open **http://localhost:3000** (UI). API docs at **http://localhost:8000/docs**.

### Stop

```bash
lsof -ti:3000,8000 | xargs kill
```

---

## What the flags mean

| Flag | Command | Meaning |
|------|---------|---------|
| `-y` | `conda create … -y` | "Yes to all" — auto-confirms the prompt so it runs unattended. |
| `-e` | `pip install -e <path>` | **Editable** install — links to the source folder instead of copying it in. Edits take effect immediately; deleting the folder breaks the install. |
| `-r` | `pip install -r <file>` | **Requirements** — the arg is a *file listing packages*, not a single package name. Installs every line. |

## What actually happens at each step

- **`conda create -n lerobot python=3.10 -y`** — builds a brand-new **isolated** Python env in `~/miniforge3/envs/lerobot/` with its own Python 3.10 + pip. Nothing installed system-wide; no conflicts with other projects.
- **`conda activate lerobot`** — points your shell's `python`/`pip` at that env until you `conda deactivate` or close the terminal.
- **`pip install -e lerobot-MakerMods`** — installs lerobot's deps (torch, transformers, motor libs…) into the env, drops an **editable link** to the source folder, and creates the CLI tools the app shells out to: `lerobot-record`, `lerobot-calibrate`, `lerobot-teleoperate` (in `…/envs/lerobot/bin/`).
- **`pip install -r MakerMods-App/requirements.txt`** — installs the backend's own deps (`fastapi`, `uvicorn`, `opencv-python-headless`, `huggingface_hub`) into the same env, so the server and the robot engine share one interpreter.
- **`npm install`** — reads `frontend/package.json`, downloads JS deps into `frontend/node_modules/`. The Node equivalent of "create env + install", but per-folder, not global.

## Mental model

```
conda create   ->  build an empty, isolated Python sandbox
conda activate ->  point your shell's python/pip at that sandbox
pip install -e ->  install the robot engine, LINKED to the folder (live, not copied)
pip install -r ->  install the app's server deps from a list, into the same sandbox
npm install    ->  same idea, but for the JS frontend (into node_modules/)
```

`conda` manages the **environment** (interpreter + a place to put packages);
`pip`/`npm` **fill it with packages**.

---

## Gotchas

- **`lerobot-MakerMods` is load-bearing.** It's installed `-e` (editable/linked), so the
  app imports and runs code straight from that folder. Don't move or delete it — the
  install breaks and the backend won't start. The CLI tools (`lerobot-record`, etc.) and
  `import lerobot` both resolve into `lerobot-MakerMods/src/lerobot`.
- **The app needs the fork, not upstream `lerobot/`.** It builds commands like
  `--robot.type=so101_follower` / `bi_so101_follower` and uses Feetech auto-calibration +
  Qualia training hooks that only exist in the fork (v0.3.4). The upstream `lerobot/` folder
  (v0.5.2) uses different robot names and is NOT what the app runs against.
- **Robot ports** live in `MakerMods-App/webui_config.json` (macOS `/dev/cu.usbmodem…` paths).
  These can change on reboot/replug; if the app "can't find the robot," recheck them first.
  The UI still loads fine without hardware — only setup/calibrate/teleop will error.
- **Training is remote.** The training tab calls the **Qualia** cloud GPU service, not your
  local machine.
