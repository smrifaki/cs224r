"""
Modal app entry-points.

Smoke test - run after every image/dependency change:
    modal run infra/modal_app.py

Checks:
  - Image builds, all packages import, foveated package is installed
  - GPU is available and ViT-small forward pass succeeds
  - Modal Volumes mount correctly
  - wandb secret is wired up and a test run logs successfully
"""
from modal_config import app, ckpt_vol, data_vol, hf_secret, image, wandb_secret


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[hf_secret, wandb_secret],
    timeout=300,
)
def smoke() -> None:
    import os
    from pathlib import Path

    import timm
    import torch
    import wandb

    os.chdir("/root/project")

    # --- foveated package ---
    import foveated
    print(f"{foveated.__name__} package imported OK")

    # --- GPU ---
    print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "T4 GPU not detected - check Modal GPU config"

    # --- ViT-small forward pass ---
    model = timm.create_model("vit_small_patch16_224", pretrained=True).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ViT-small params: {n_params:.1f}M")

    x = torch.randn(4, 3, 224, 224, device="cuda")
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1000), f"unexpected output shape {out.shape}"
    print(f"forward pass OK - logits shape: {list(out.shape)}")

    # --- Volumes ---
    print(f"/data: {list(Path('/data').iterdir()) or '(empty)'}")
    print(f"/ckpt: {list(Path('/ckpt').iterdir()) or '(empty)'}")

    # --- wandb ---
    run = wandb.init(
        project="cs224r-foveated",
        job_type="smoke",
        tags=["smoke"],
        config={"gpu": "T4", "vit_params_M": round(n_params, 1)},
    )
    wandb.log({"smoke/forward_pass_ok": 1})
    wandb.finish()
    print(f"wandb run: {run.url}")

    print("\nModal smoke test PASSED")


@app.local_entrypoint()
def main() -> None:
    smoke.remote()
