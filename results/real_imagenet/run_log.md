# Real ImageNet-100 PPO baseline v2 — proposal-quality

Six T4 jobs in parallel via `infra/modal_real_imagenet_v2.py::main`:
2 agents (A, D) x 3 seeds (0, 1, 2), 60,000 env-steps each.

## Setup

* Modal T4, 100-class subset of ImageNet sampled from the HF
  dataset on the `cs224r-imagenet` Modal Volume.
* Backbone: timm `vit_small_patch16_224` (frozen pretrained
  weights, 384-d features). A fresh 100-class linear head is fine-
  tuned for 2 epochs on whole-image features.
  * Backbone whole-image val accuracy: 0.888 - 0.902 across seeds.
* Foveated MDP: 7x7 patch grid pooled from the ViT's 14x14 patches,
  K=8 reveals per episode, patch_cost = 0.01.
* PPO via stable-baselines3, 60k env-steps, n_steps=512, batch=64,
  lr=3e-4, MLP policy [256, 256].

## Per-(agent, seed) eval

| agent | seed | backbone val acc | eval acc | train_last500 acc | wall time |
|-------|-----:|-----------------:|---------:|------------------:|----------:|
| A     | 0    |  0.888           |  0.843   |  0.900            |  136 s    |
| A     | 1    |  0.902           |  0.875   |  0.912            |  134 s    |
| A     | 2    |  0.902           |  0.810   |  0.896            |  143 s    |
| D     | 0    |  0.888           |  0.840   |  0.894            |  136 s    |
| D     | 1    |  0.902           |  0.873   |  0.900            |  141 s    |
| D     | 2    |  0.902           |  0.805   |  0.900            |  136 s    |

Aggregate over 3 seeds:

| agent | eval acc mean | eval acc std |
|-------|--------------:|-------------:|
| A     |  0.843        |  0.027       |
| D     |  0.839        |  0.028       |

Random baseline for 100 classes: 0.010. The K=8 PPO agents land
within ~5% of the backbone's whole-image accuracy (~0.89), so the
foveated policy has learned to commit informative patches.

Agents A and D are at parity. The proposal predicts C > A, D >= A,
B > A; we still need Agents B (intrinsic-reward bonus) and C
(residual-as-feature) which both require a pretrained forward-
dynamics model on (z_t, action, z_{t+1}) triples. That dynamics
pretrain is the next step (~30 min on T4 for the dataset + ~10
min model train).

## Fix vs v1

v1 (sweep_A_D_seeds_0_1_2.json) capped at 0.02-0.04 eval acc
because the foveated env's obs pooled only the first 8 of 384 ViT
feature dims and then zero-padded back to 384 for the classifier.
v2 keeps the obs vector compact (mask + per-patch residual +
assembly entropy = 99-d) and gives the classifier the **full**
384-d mean-pooled feature over the committed mask. The factor-25x
jump from 0.03 -> 0.84 confirms the bottleneck was that pooling.

## Run

```bash
modal run infra/modal_real_imagenet_v2.py::main \
  --agents A,D --seeds 0,1,2 --n-env-steps 60000 --n-classes 100
```
