# Real ImageNet-100 PPO sweep on Modal T4

Six T4 jobs in parallel via `infra/modal_real_imagenet.py::main`:
2 agents (A, D) x 3 seeds (0, 1, 2), 50,000 env-steps each.

## Setup

* Modal T4, 100-class subset of ImageNet sampled from the HF
  dataset on the `cs224r-imagenet` Modal Volume.
* ViT-small (timm `vit_small_patch16_224`) loaded with pretrained
  weights; a fresh 100-class linear head fine-tuned for 1 epoch on
  whole-image features.
  * Backbone head val acc on 100-class ImageNet: **0.878**.
* Foveated MDP: 7x7 patch grid, K=8 reveals per episode.

## Per-job results

| agent | seed | eval acc | wall time |
|-------|-----:|---------:|----------:|
| A     | 0    |  0.040   |   101 s   |
| A     | 1    |  0.020   |    99 s   |
| A     | 2    |  0.035   |   111 s   |
| D     | 0    |  0.040   |   124 s   |
| D     | 1    | (running)|     -     |
| D     | 2    |  0.030   |   124 s   |

Random baseline on 100 classes = 0.01. Agents are ~3x random but
~25x below the backbone's whole-image accuracy (0.878).

## What this is

Real Modal-T4 PPO on real ImageNet-100. Reproducible by running
`infra/modal_real_imagenet.py::main` against a clean Modal
workspace (HF dataset on `cs224r-imagenet` volume, wandb secret).

## Why the accuracy is low

The env's observation pools the first 8 of 384 ViT feature dims to
keep the obs vector tractable for stable-baselines3 PPO. The
classifier head then receives a zero-padded 384-dim vector with only
8 dims set, so the classifier collapses to ~uniform over classes
once the agent reveals patches. The PPO learns *which* patches to
reveal but the downstream head can't translate that into class
margin.

## Fix path (not yet shipped)

Two changes restore proposal-quality:

1. Decouple the observation (kept compact at 49 + 8 + 49 dims) from
   the classifier (re-uses the full 384-dim per-patch features
   pooled over the committed mask).
2. Train at K-sweep {4, 8, 16} with 200k env-steps per (agent, seed)
   and add a forward-dynamics-model pretrain pass so Agents B + C
   can run too.

Estimate: ~6 hours additional Modal-T4 compute.
