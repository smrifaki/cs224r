"""
Shared Modal definitions: image, app, secrets, volumes.
Imported by sibling infra scripts as:
    from modal_config import app, image, ...

Before running any function, create secrets once:
    modal secret create huggingface HF_TOKEN=hf_...
    modal secret create wandb WANDB_API_KEY=...
"""
import modal

# Dependency layer - rebuilt only when requirements.txt or apt packages change.
_deps = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libmagickwand-dev", "git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("imagenet-c")
)

# Source layer goes last so the expensive dep layers stay cached when code changes.
# add_local_dir copies the whole project root into the image at build time.
# pip install -e makes `import foveated` work inside every function.
image = (
    _deps
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["data/", "wandb/", ".git/", "__pycache__", "*.egg-info"],
        copy=True,
    )
    .run_commands("pip install -e /root/project")
)

app = modal.App("cs224r-foveated", image=image)

# Create secrets once via:
#   modal secret create huggingface HF_TOKEN=<your-hf-read-token>
#   modal secret create wandb WANDB_API_KEY=<your-wandb-key>
hf_secret = modal.Secret.from_name("huggingface")
wandb_secret = modal.Secret.from_name("wandb")

# /data   -  ImageNet + ImageNet-C (treat as read-only during training)
# /ckpt   -  model checkpoints (read-write)
data_vol = modal.Volume.from_name("cs224r-imagenet", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("cs224r-ckpts", create_if_missing=True)
