# Results

All numbers below come from real Modal compute runs. No synthetic-
decision-layer simulations anywhere in this directory.

## ImageNet-100 PPO sweep (proposal-quality)

Full 4-agent × 3-seed sweep with forward-dynamics pretrain on
Modal T4. Real HF ImageNet-100, real pretrained ViT-small backbone,
real PPO via stable-baselines3.

| agent | mechanism                          | mean eval acc |
|-------|------------------------------------|--------------:|
| A     | baseline                           |    0.845      |
| B     | intrinsic reward = β · ‖residual‖² |    0.823      |
| C     | precision-weighted residual obs    |    0.833      |
| D     | assembly entropy obs               |    0.836      |

Backbone whole-image val acc: 0.882–0.901. Random baseline 0.010.
Agents recover **~94% of the backbone ceiling** while seeing only
K=8 of 49 patches.

Per-seed numbers + dynamics pretrain history in
[results/real_imagenet/](results/real_imagenet/).

## CIFAR-10 PPO smoke (validation that the stack runs end-to-end)

A 28-second T4 PPO run on CIFAR-10 with a 1-epoch CNN backbone.
Eval acc 0.135 vs 0.10 random; the backbone capped at 0.371 so the
agent has very little headroom. Kept only to show that the full
PPO + foveated-env code path runs on real images.

In [results/real_ppo/](results/real_ppo/).

## What's missing (for the full proposal)

* ImageNet-1k (1000 classes) instead of the 100-class subset.
* ImageNet-C corruptions (15 types) for the distribution-shift eval
  that the C > A claim depends on.
* K-sweep over {4, 6, 8, 10, 12, 16} for the Pareto curve.
* Held-out corruption split, paired permutation tests on regret.

Each is a sweep extension, not a rewrite. The infrastructure
(`infra/modal_real_full.py`) takes config flags.
