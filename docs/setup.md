# Setup: Local + Modal Environments and Data

Goal: a reproducible local dev environment, a matching Modal image, and ImageNet/ImageNet-C accessible from both. By the end of this doc you should be able to run a 1-minute smoke training loop locally and on Modal.

This is the prerequisite for [phase1_environment.md](phase1_environment.md).

---

## 0. Decisions locked here

- **Python:** 3.11 (matches Modal's current default and most recent `timm` wheels).
- **PyTorch:** CUDA 12.1 build to match Modal's NVIDIA images.
- **Compute home:** Modal for all multi-hour runs. Local box is for environment debugging and figure generation only.
- **Tracking:** wandb, project `cs224r-foveated`.
- **Data home:** Modal Volume `cs224r-imagenet` (read-only mounted at train time). Local box keeps only a tiny smoke subset under `data/smoke/`.

---

## 1. Local environment

### 1.1 Create the conda env

```bash
conda create -n cs224r-final python=3.11 -y
conda activate cs224r-final
```

### 1.2 Install PyTorch (must match Modal CUDA)

```bash
pip install torch==2.4.* torchvision==0.19.* --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you are on a laptop with no CUDA-capable GPU, install the CPU wheels instead - local is just for smoke tests:
```bash
pip install torch==2.4.* torchvision==0.19.* --index-url https://download.pytorch.org/whl/cpu
```

### 1.3 Project dependencies

Maintain these in `requirements.txt` at the repo root. Initial contents:

```
# core ML
timm>=1.0.0
transformers>=4.44
datasets>=3.0
huggingface_hub>=0.24

# RL / env
gymnasium>=1.0
numpy<2.0          # gymnasium 1.0 still has rough edges with numpy 2
scipy>=1.11

# infra
modal>=0.64
wandb>=0.17
pyyaml>=6.0
tqdm
typer              # CLI for train/eval entry points

# plotting & analysis
matplotlib
seaborn
pandas

# dev
pytest
pytest-xdist
ruff
```

Install:
```bash
pip install -r requirements.txt
```

### 1.4 Editable install of the project package

After scaffolding `src/foveated/` (see [phase1_environment.md](phase1_environment.md)), make it importable:

```bash
pip install -e .
```

`pyproject.toml` minimum:
```toml
[project]
name = "foveated"
version = "0.0.1"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]
```

### 1.5 Sanity script

Drop this as `scripts/smoke_env.py` and run it after each setup change:

```python
import torch, timm, gymnasium, modal, wandb, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("timm", timm.__version__)
print("gymnasium", gymnasium.__version__)
print("modal", modal.__version__)
m = timm.create_model("vit_small_patch16_224", pretrained=True)
print("vit params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
```

---

## 2. HuggingFace access

ImageNet-1k is gated. Steps:

1. Create / log in to HF account.
2. Visit `https://huggingface.co/datasets/ILSVRC/imagenet-1k` and request access. Wait for approval.
3. `huggingface-cli login` locally; paste a read token.
4. For Modal, store the same token as a Modal Secret named `huggingface` with key `HF_TOKEN`.

ImageNet-C: there are several mirrors on HF. The cleanest path is to **generate corruptions on the fly** from clean ImageNet using the published `imagenet_c` Python package - that avoids the 75GB download. Use the precomputed dataset only if reproducibility against published numbers is critical.

```bash
pip install imagenet-c
```

Note: `imagenet-c` depends on `scikit-image` and `wand` (ImageMagick bindings). On Windows, ImageMagick install is finicky - easiest path is to do the corruption work **only on Modal** (Linux), and keep local smoke tests on clean images. Document this in `docs/design_decisions.md` if we go that route.

---

## 3. Modal setup

### 3.1 Auth

```bash
modal token new
```

### 3.2 Image definition

`infra/modal_image.py`:

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libmagickwand-dev", "git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("imagenet-c")
    .add_local_python_source("foveated")  # mounts src/foveated into image
)

app = modal.App("cs224r-foveated", image=image)

hf_secret = modal.Secret.from_name("huggingface")
wandb_secret = modal.Secret.from_name("wandb")
```

Pin versions in `requirements.txt` (not here) so local and Modal stay in lockstep.

### 3.3 Volume for data + checkpoints

```python
data_vol = modal.Volume.from_name("cs224r-imagenet", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("cs224r-ckpts", create_if_missing=True)
```

Mount points (conventions, used everywhere):
- `/data` → `data_vol` (read-only at train time)
- `/ckpt` → `ckpt_vol` (read-write)

### 3.4 One-shot data download

`infra/download_data.py`: a Modal function that streams clean ImageNet train+val from HF into `/data/imagenet/`. Run once:

```bash
modal run infra/download_data.py
```

Keep the script idempotent - it should skip files that already exist. Expected runtime: 1-2 hours, mostly bandwidth.

### 3.5 GPU choice

- Smoke tests: `gpu="T4"` (cheap).
- Real training: `gpu="A10G"` for Agents A/C, `gpu="A100"` only if memory is tight.
- One env vector across one GPU is plenty for the foveated env; don't bother with multi-GPU.

### 3.6 Smoke Modal job

`infra/modal_app.py` should have a `smoke()` function that loads the ViT, runs 16 random-policy episodes in the env, logs to wandb, and exits. Run it on every code change before launching long jobs:

```bash
modal run infra/modal_app.py::smoke
```

---

## 4. wandb

```bash
wandb login
```

Create the project `cs224r-foveated` (web UI). Store the API key as a Modal Secret `wandb` with key `WANDB_API_KEY`.

Default run name convention: `agent{A|B|C}_seed{n}_{git_short_sha}_{timestamp}`. Tag every run with `agent`, `seed`, `phase`.

---

## 5. Repo hygiene

`.gitignore` additions:
```
runs/
data/
*.ckpt
wandb/
.venv/
__pycache__/
```

`scripts/smoke_local.py` runs the env smoke + a 100-step PPO update locally with a tiny ViT-tiny to exercise the wiring without needing CUDA. Use it as the pre-push check:

```bash
python scripts/smoke_local.py
```

---

## 6. Gate to start Phase 1

You are ready to start [phase1_environment.md](phase1_environment.md) when:

- [ ] `python scripts/smoke_env.py` runs locally without error and reports a CUDA device (or CPU on a laptop).
- [ ] `modal run infra/modal_app.py::smoke` runs end-to-end and produces a wandb run.
- [ ] `modal run infra/download_data.py` has populated `/data/imagenet/` on the Volume.
- [ ] HuggingFace token works both locally and via Modal secret.
- [ ] `pip install -e .` succeeds and `import foveated` works from a Python REPL.
