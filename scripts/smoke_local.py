"""
Pre-push smoke test - CPU only, no GPU, no network.
Run from the project root:
    python scripts/smoke_local.py

This script tests only what exists right now (setup phase).
Add sections as Phase 1+ modules land.
"""
import sys
import traceback


def ok(label: str) -> None:
    print(f"  [OK] {label}")


def fail(label: str, exc: Exception) -> None:
    print(f"  [FAIL] {label}: {exc}")
    traceback.print_exc()
    sys.exit(1)


print("=" * 50)
print("smoke_local.py")
print("=" * 50)

# ── 1. Package imports ────────────────────────────────
print("\n[1/4] package imports")
try:
    import torch
    import timm
    import gymnasium
    import wandb
    import datasets  # noqa: F401

    ok("torch, timm, gymnasium, wandb, datasets")
except Exception as e:
    fail("package imports", e)

try:
    import foveated  # noqa: F401

    ok("foveated (editable install)")
except Exception as e:
    fail("foveated import - did you run `pip install -e .`?", e)

# ── 2. ViT-tiny CPU forward pass ──────────────────────
print("\n[2/4] ViT-tiny forward pass (CPU, random weights)")
try:
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1000), f"got {tuple(out.shape)}"
    ok(f"output shape {list(out.shape)}")
except Exception as e:
    fail("ViT-tiny forward pass", e)

# ── 3. Basic gymnasium env ────────────────────────────
print("\n[3/4] gymnasium CartPole smoke (placeholder)")
try:
    env = gymnasium.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,), f"unexpected obs shape {obs.shape}"
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    ok("CartPole step OK")
except Exception as e:
    fail("gymnasium CartPole", e)

# ── 4. Foveated env (Phase 1 placeholder) ─────────────
print("\n[4/4] foveated env (skipped - Phase 1 not yet implemented)")
# Uncomment once Phase 1 is done:
# from foveated.envs import FoveatedEnv, make_vec_env
# env = FoveatedEnv(split="smoke")
# obs, _ = env.reset()
# env.close()
# ok("FoveatedEnv step OK")
print("  [--] skipped")

print("\n" + "=" * 50)
print("All checks passed.")
print("=" * 50)
