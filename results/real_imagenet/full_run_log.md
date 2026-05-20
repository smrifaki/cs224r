# Full 4-agent x 3-seed proposal sweep on ImageNet-100

Twelve T4 PPO jobs + one A10G dynamics pretrain via
`infra/modal_real_full.py::main`.

## Setup

* Modal T4, 100-class subset of ImageNet sampled from the HF
  dataset on the `cs224r-imagenet` Modal Volume.
* Backbone: timm `vit_small_patch16_224` (frozen). Per-seed head
  fine-tune for 2 epochs on whole-image features.
* Dynamics pretrain: ~4000 random rollouts on the foveated env
  yielded ~32000 (z_t, action, z_{t+1}) triples; trained a 256-h
  MLP with Gaussian (mean + log-var) head for 30 epochs.
* Foveated MDP: 7x7 patch grid pooled from the ViT's 14x14
  patches, K=8 reveals per episode, patch_cost = 0.01.
* PPO via stable-baselines3, 60k env-steps, MLP [256, 256].

## Headline table (eval acc on held-out val set, 400 episodes)

| agent | mechanism                          | seed 0 | seed 1 | seed 2 | mean  |
|-------|------------------------------------|-------:|-------:|-------:|------:|
| A     | baseline (no extra channel)        | 0.835  | 0.863  | 0.838  | 0.845 |
| B     | intrinsic reward = beta x residual | 0.840  | 0.823  | 0.807  | 0.823 |
| C     | precision-weighted residual obs    | 0.810  | 0.850  | 0.840  | 0.833 |
| D     | assembly entropy obs               | 0.797  | 0.858  | 0.853  | 0.836 |

Backbone whole-image val acc: 0.882 - 0.901. Random: 0.010. The
agents recover **~94%** of the backbone's whole-image accuracy
while seeing only K=8 of the 49 patches.

## What the agents are doing

All 4 land within seed-noise of each other on clean ImageNet-100.
The proposal's "C > A" claim is about adaptation under
distribution shift — clean ImageNet-100 does not include shift. To
test the claim we need:

1. Synthesise the 15 ImageNet-C corruption types via the
   `imagenet-c` pip package.
2. Eval each (agent, seed) checkpoint on the held-out corruptions.
3. Compare per-agent regret to the backbone whole-image ceiling.

That's the **next** sweep, queued.

## Run

```bash
modal run infra/modal_real_full.py::main \
  --agents A,B,C,D --seeds 0,1,2 --n-env-steps 60000 --n-classes 100
```
