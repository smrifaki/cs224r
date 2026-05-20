# Reproduce

The headline numbers in [RESULTS.md](RESULTS.md) come from real
Modal-T4 PPO training on real HF ImageNet-100, with a forward-
dynamics pretrain on Modal A10G. Twelve PPO jobs run in parallel
on T4 (4 agents × 3 seeds).

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install modal numpy matplotlib
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
modal run infra/modal_real_full.py::main \
  --agents A,B,C,D --seeds 0,1,2 --n-env-steps 60000 --n-classes 100
```

Default: 100-class subset of ImageNet sampled deterministically from
class indices via `np.random.default_rng(0)`, 60 train + 12 val
images per class. The backbone is timm `vit_small_patch16_224`
(frozen pretrained weights); a 100-class head is fine-tuned for 2
epochs on whole-image features per (agent, seed). The 7×7 foveated
grid is pooled from the ViT's 14×14 patch tokens; K=8 reveals per
episode.

Headline checks after a run:

* `results/real_imagenet/sweep_full_A_B_C_D_seeds_0_1_2.json`
  contains 12 rows; mean `eval_mean_acc` per agent should land in
  0.80–0.86.
* `results/real_imagenet/sweep_v2_A_D_seeds_0_1_2.json` is the v2
  baseline (A and D only) that demonstrated the fix to the
  feature-pooling bug; mean eval acc 0.84.
* `train_mean_acc_last500` on each (agent, seed) shows the train-
  time accuracy plateau, ~0.89–0.92.

The first PPO smoke ever ran on CIFAR-10 at much smaller scale:

* `results/real_ppo/smoke_seed0.json` — 12k env-steps, eval acc
  0.135 (vs 0.10 random), backbone capped at 0.371. Kept only to
  prove the full stack runs end-to-end on real images.
