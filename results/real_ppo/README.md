# Real PPO runs

The numbers in `results/` (one level up) are from the decision-layer
simulation, byte-identical across re-runs. The artefacts here are
**real Modal-T4 PPO training jobs** on real CIFAR-10 images.

## smoke_seed0.json

* T4 GPU, 12k env-steps, 28.5s wall-clock.
* Backbone: a 3-conv head trained for 1 epoch on whole-image
  CIFAR-10, ending at 37.1% val accuracy.
* Foveated env: 4x4 patch grid over the 32x32 CIFAR images, K=8
  reveals per episode, patch_cost = 0.005.
* Eval over 200 deterministic-policy episodes:
  * `eval_mean_acc`     = 0.135 (vs 0.10 random baseline)
  * `eval_mean_reward`  = -0.014

Interpretation: at this training budget the agent is only marginally
above chance. The bottleneck is the backbone (37% val acc cap means
even a perfect oracle reveal pattern would not exceed 0.37). Real
runs need either a stronger backbone (a pre-trained ViT-small as
the proposal calls for) or significantly more PPO steps. The
smoke run validates the full RL stack end-to-end.

Larger runs land in `large_seedN.json` as they complete.
