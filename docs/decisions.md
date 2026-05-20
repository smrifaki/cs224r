# Design decisions

Running log of the choices that shape the real Modal ImageNet-100
sweep and the headline numbers in [RESULTS.md](../RESULTS.md). One
bullet per decision: what was decided, the alternatives that were
considered, the reasoning.

## Real ImageNet PPO sweep

* **100-class subset of ImageNet, deterministic class draw.** Full
  1000-class ImageNet is the proposal target but the 100-class
  subset trains in ~140 s per (agent, seed) on T4 and is enough to
  establish the relative agent ordering. The class set is fixed
  via `np.random.default_rng(0).choice(1000, 100)` so re-runs are
  comparable.
* **60 train / 12 val images per class.** 6 k train + 1.2 k val
  fits in T4 RAM with the cached ViT-small per-patch features.
* **K = 8 reveals out of 49 patches.** Matches the proposal lock.
  Smaller K loses too much accuracy on whole-image; larger K leaves
  no headroom over the backbone.
* **60 k env-steps per PPO job.** Empirically the train-time
  accuracy plateaus by ~30 k; 60 k buys a small extra margin.

## Backbone + head

* **timm `vit_small_patch16_224`, pretrained, frozen.** 384-d
  features, 196 patch tokens. Fine-tuning the backbone is the
  ImageNet-C distribution-shift step and is deferred.
* **2-epoch linear head on whole-image features.** 100-class val
  acc lands at 0.88–0.90.

## Foveated MDP design

* **7×7 patch grid pooled from 14×14 ViT tokens.** 2×2 mean pool
  over the ViT patch tokens; cheap and keeps the 7×7 grid the
  proposal specifies.
* **Decoupled obs / classifier paths.** The PPO observation is
  compact (mask + per-patch residual + assembly entropy = 99 d).
  The classifier head receives the **full 384-d mask-pooled
  feature**, not a zero-padded slice. v1 of the env compressed
  features to 8 d before classification and capped accuracy at
  ~3 %; v2 with this decoupling jumps to ~84 %.

## Agent definitions

* **A: baseline.** No extra channel in the obs.
* **B: intrinsic reward** = β · ‖residual‖² with β = 0.05.
  Residual computed via the dynamics model when present, else the
  ‖feat − running mean‖² fallback.
* **C: precision-weighted residual obs.** Per-patch dynamics-model
  residual fed into the observation vector. Uses the pretrained
  dynamics model.
* **D: assembly entropy obs.** Classifier entropy of the current
  reveal, broadcast across patches.

## Forward-dynamics pretrain

* **Random rollouts (~32 k triples) on the foveated env, dynamics
  trained for 30 epochs.** Gaussian (mean + log-σ) output head.
  Trained once before the PPO sweep and frozen for B/C training.

## Statistical comparisons (deferred)

* The proposal's headline claim is **C > A under distribution
  shift**. On clean ImageNet-100 the four agents are at parity
  within seed noise. The next sweep adds the 15 ImageNet-C
  corruption types via the `imagenet-c` pip package and a
  held-out corruption split for paired permutation testing.
