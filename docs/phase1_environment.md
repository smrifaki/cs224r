# Phase 1: Foveated Environment + Backbone (Weeks 1-2)

Goal: a working `gymnasium.Env` that wraps a frozen ViT-small classifier and supports the foveated-classification MDP. The environment is the substrate everything else (PPO, dynamics model, evaluation) sits on, so it pays to over-test it here before moving on.

Prerequisite: [setup.md](setup.md) gates are all green.
Next: [phase2_ppo_baseline.md](phase2_ppo_baseline.md).

---

## 1. MDP definition (locked here)

| element | value |
|---|---|
| Image input | 224×224 RGB, ImageNet preprocessing (mean/std) |
| Patch grid | **7×7 = 49 patches** (each patch is 32×32 px). Revisit if action space is the bottleneck. |
| Low-res view | bilinear downsample to 56×56, then upsample back to 224 so the ViT input shape is constant. |
| Action space | `Discrete(49)`. Fixed horizon `K=8` reveals per episode. No stop action in v1 (add as ablation). |
| Observation | `Dict({"low_res_emb": Box(d_emb,), "mask": MultiBinary(49), "goal": Box(num_corruptions,), "pred_err": Box(d_err,)})` |
| Reward | At terminal step: `-CE(logits, label) - patch_cost * num_revealed`. Per-step intermediate reward = 0 in v1 (matches Agent A baseline). |
| Episode length | 8 steps (fixed). |
| Reset | sample (image, label, corruption_id, severity) from current dataset distribution. |

`d_emb` is the ViT-small pooled feature dim (= 384). `d_err` is 0 for Agents A/B and per-patch residual size for Agent C (filled in in Phase 3); for now reserve the slot but populate with zeros.

`patch_cost` default: `0.02`. Tune so that the baseline policy converges to revealing roughly half the patches on clean data.

### Why these choices
- **7×7 vs 14×14:** ViT-small's native patch grid is 14×14, but PPO with 196 discrete actions and only 8 reveals per episode wastes most actions. Coarser grid keeps the action distribution learnable. We're accepting the loss of native alignment in exchange.
- **Fixed horizon:** learned stop-actions add training instability that is not part of the research question. We can add it as an ablation in Phase 5 if time allows.
- **Low-res-upsampled vs native low-res:** keeps the ViT input pipeline shape-stable, which means we can re-use the pretrained head with no surgery.

---

## 2. Files to create

```
src/foveated/
├── data/
│   ├── __init__.py
│   ├── imagenet.py
│   ├── imagenet_c.py
│   ├── transforms.py
│   └── splits.py
├── envs/
│   ├── __init__.py
│   ├── foveated_env.py
│   ├── vec_env.py
│   └── classifier_head.py
├── models/
│   ├── __init__.py
│   └── backbone.py
└── utils/
    ├── __init__.py
    ├── seeding.py
    └── logging.py
tests/
├── test_backbone.py
├── test_env.py
└── test_data.py
configs/
└── env.yaml
```

---

## 3. Step-by-step build

### Step 1.1 - Backbone wrapper (`models/backbone.py`)
- Function `load_vit_small(pretrained=True)` returns a frozen `timm` `vit_small_patch16_224` plus a `forward_features(x)` helper that returns the pooled CLS embedding (shape `(B, 384)`).
- Function `embed_patches(model, x)` returns the per-patch tokens too (shape `(B, 196, 384)`) - needed later by the dynamics model. Average-pool 4 native ViT patches into 1 "macro-patch" so the 7×7 grid lines up with 49 macro-patches.
- Set `requires_grad=False` on all backbone params. Move to `device` in a single helper.

**Test (`test_backbone.py`):** loading is deterministic; pooled-feature shape is correct; top-1 accuracy on a 256-image ImageNet val subset matches `timm`'s published number within 0.5pp.

### Step 1.2 - Data wrappers (`data/imagenet.py`, `data/imagenet_c.py`)
- `imagenet.py`: `ImageNetDataset(split, root)` over the Modal Volume path. Yields `(image_tensor_224, label_int)`.
- `imagenet_c.py`: wraps `imagenet-c` to apply corruption-and-severity on the fly. Yields `(image_tensor_224, label_int, corruption_id, severity)`. Corruptions are enumerated in `splits.py` (fixed order, 15 entries + clean = index 15).
- `transforms.py`: ViT preprocessing pipeline, plus `make_low_res(image)` and `extract_patch(image, idx, grid=7)` primitives.
- `splits.py`: lists train corruptions and held-out corruptions; this file is the source of truth and is committed. Default split (commit unless overridden in Section 6 below):
  - **Train (10):** gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog
  - **Held-out (5):** brightness, contrast, elastic_transform, pixelate, jpeg_compression
  - **Severities trained:** 1-3. Severities 4-5 held out for stress eval.

**Test (`test_data.py`):** shape and dtype assertions; a fixed seed produces a fixed (image_id, corruption, severity) tuple; corruption application is idempotent given the same seed.

### Step 1.3 - Classifier head (`envs/classifier_head.py`)
The pretrained ViT-small head expects full-res images, not partially foveated ones. We need to either (a) freeze the head and accept some accuracy loss on foveated views, or (b) fine-tune a fresh head on synthetic foveated views.

Decision for v1: **(b) - fine-tune a fresh head on synthetic foveated views.** The cost is a one-shot pretraining pass; the benefit is a much higher reward ceiling, which is what the agent learns to chase.

- `ClassifierHead(d_emb=384, n_classes=1000)`: 2-layer MLP, dropout 0.1.
- Pretraining script `scripts/pretrain_head.py`:
  - For each minibatch: sample a clean image, sample a random `k ∈ [0, 49]` patches to reveal, build the foveated input, forward through frozen backbone, train head with cross-entropy.
  - Train for ~1 epoch over ImageNet train. Save to `/ckpt/heads/foveated_head_clean.pt`.
- This head is **fixed** during RL training (further fine-tuning during RL would let the policy and the head co-adapt in confounding ways).

**Smoke check:** trained head should beat the pretrained head's accuracy when evaluated on random foveated views; it should also still reach >70% top-1 when given the full image.

### Step 1.4 - The Env itself (`envs/foveated_env.py`)
Class `FoveatedEnv(gymnasium.Env)`:

```python
def __init__(self, dataset, backbone, head, patch_cost=0.02, horizon=8, grid=7, device="cuda"):
    ...
    self.action_space = gym.spaces.Discrete(grid * grid)
    self.observation_space = gym.spaces.Dict({
        "low_res_emb": Box(low=-inf, high=inf, shape=(384,)),
        "mask":        MultiBinary(grid * grid),
        "goal":        Box(low=0, high=1, shape=(16,)),   # 15 corruptions + clean
        "pred_err":    Box(low=-inf, high=inf, shape=(D_ERR,)),  # zeros in v1
    })
```

Internal state per episode:
- `self.image_full` (224×224)
- `self.image_current` (224×224, starts as low-res-upsampled version)
- `self.mask` (49 bools, all False at reset)
- `self.label`, `self.corruption_id`, `self.severity`
- `self.steps_taken`

`reset()`:
1. Sample one item from the dataset.
2. Build low-res view; cache it.
3. Compute pooled embedding of low-res view through frozen backbone (call this `low_res_emb`).
4. Initialize `mask` to zeros, `pred_err` to zeros, `steps_taken=0`.
5. Return observation dict.

`step(action)`:
1. If `mask[action]` already True, no-op cost (return small negative reward `-2 * patch_cost` to discourage). Don't increment `steps_taken`. (Alternative: mask invalid actions in policy - see Phase 2.)
2. Else: copy patch from `image_full` into `image_current` at `action`'s grid slot; flip `mask[action]`; increment `steps_taken`.
3. Recompute `low_res_emb` from `image_current` (yes, every step - cost is one ViT forward, fine on GPU).
4. **Pred_err update** (placeholder in Phase 1, filled in by Phase 3): zeros.
5. If `steps_taken == horizon`: terminate, compute reward = `-CE(head(low_res_emb), label) - patch_cost * mask.sum()`. Else reward = 0.
6. Return obs, reward, terminated, truncated, info. Info includes `predicted_class`, `correct`, `num_revealed`.

**Tests (`test_env.py`):**
- Episode length is exactly `horizon`.
- Revealing all 49 patches in sequence reaches accuracy within 0.5pp of the pretrained-head accuracy on full images (sanity that foveation reconstructs the original).
- Reward is 0 except on the last step.
- Reset → step trajectory is deterministic given a fixed seed.
- Repeating an already-revealed patch is penalized and does not advance state.

### Step 1.5 - Vectorized env (`envs/vec_env.py`)
A simple batched wrapper that holds N envs and steps them all on GPU: instead of `N` parallel processes, store `N` `image_current` tensors stacked along batch dim, run one big ViT forward per step.

- API: `reset()` → batched obs dict (each value has leading dim N). `step(actions)` → batched obs, rewards, dones.
- Reuses `FoveatedEnv` for spec but doesn't inherit; cleaner to have two implementations than to fight `gymnasium.vector`.

**Test:** vectorized rollout matches the per-env rollout when given identical seeds and actions.

### Step 1.6 - Smoke training loop (no PPO yet)
`scripts/smoke_random.py`:
- Build env, run 1000 random-policy episodes on clean ImageNet, log mean accuracy and mean patches revealed per episode to wandb.
- Should report something like ~50% top-1 (random patches usually give the head enough info on most images) and exactly `horizon` reveals (since random policy uses every step).

Run it both locally (with a tiny ViT-tiny if no GPU) and on Modal.

---

## 4. Wandb logging during Phase 1

Per-episode metrics:
- `env/accuracy_per_episode` (0 or 1)
- `env/reward_terminal`
- `env/num_revealed`
- `env/corruption_id`
- `env/severity`

Histograms:
- `env/action_distribution` over a logging window
- `env/patches_revealed_per_episode`

Sanity plots auto-generated after the smoke run:
- accuracy vs. corruption type for the random policy on ImageNet-C training corruptions (should be lower than on clean).
- accuracy vs. number of patches revealed (should be monotonically increasing).

---

## 5. Gate to Phase 2

Move on when all of these hold:

- [ ] `test_backbone.py`, `test_env.py`, `test_data.py` all pass on Modal.
- [ ] Random policy on clean ImageNet reaches accuracy near the head's "8 random patches" reference.
- [ ] Random policy on the 10 training corruptions shows visibly lower accuracy than on clean (the regime distinction is detectable).
- [ ] Revealing all 49 patches recovers full-image accuracy within 0.5pp.
- [ ] Pretrained foveated head is saved to `/ckpt/heads/`.

---

## 6. Open questions deferred

- **Q4 (patch grid).** Commit to 7×7 for v1. Revisit before Step 1.4 if there is a prior from the toy work.
- **Q6 (held-out split).** Commit to the noise+blur+weather → digital split above. Revisit before Step 1.2.
- **Q9 (classifier head fine-tune).** Commit to fine-tuning a fresh head on synthetic foveated views. Revisit before Step 1.3.

If any of these change, the change is local to Phase 1 and does not affect later phases.
