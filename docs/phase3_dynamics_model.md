# Phase 3: Forward Dynamics Model (Week 5)

Goal: a learned forward dynamics model on the ViT-small embedding space, plus a precision-weighted residual function. This is the technical lynchpin of the project - the residual is what becomes Agent C's extra feature and what becomes Agent B's intrinsic reward bonus.

This phase is the **pairing week with Mouhssine**. The plan here is what I'd commit to absent guidance; expect to revise after the pairing session.

Prerequisite: [phase2_ppo_baseline.md](phase2_ppo_baseline.md) gates green and Agent A is training.
Next: [phase4_agents_bc.md](phase4_agents_bc.md).

---

## 1. Decisions to lock with Mouhssine (today's defaults)

| question | my default | needs confirmation? |
|---|---|---|
| Operating space | per-patch ViT token embeddings (macro-patches, 49 of them, dim 384) | **yes** |
| Conditioning | (current low_res_emb, action index, mask) → predicted next-patch embedding | yes |
| Goal-conditioning the dynamics model | **no** (would leak the regime info) | yes |
| Architecture | small transformer: 2 layers, 4 heads, d_model=256 | maybe |
| Training data | offline rollouts from a partially-trained Agent A (collected once, ~500k transitions) | yes |
| Schedule | pretrained then frozen during RL | **yes** - joint vs frozen is a substantive question |
| Precision form | per-dim running variance (one scalar per of the 384 dims) | yes |
| Residual fed to policy | precision-normalized residual vector of size 384, concatenated to obs | yes |

These map to questions Q2, Q3, Q7 in [design_decisions.md](design_decisions.md).

---

## 2. Files to create

```
src/foveated/
├── models/
│   └── dynamics.py
├── algos/
│   ├── dynamics_train.py
│   └── precision.py
├── experiments/
│   └── collect_rollouts.py
configs/
└── dynamics.yaml
tests/
├── test_dynamics.py
└── test_precision.py
```

---

## 3. Mathematical setup

### Notation
- `e_t`: pooled embedding of the current foveated image at step `t` (dim 384).
- `a_t`: action taken (patch index ∈ {0..48}).
- `p_{a_t}`: macro-patch embedding extracted at index `a_t` from the **full-resolution** image (the "ground truth" of what gets revealed).
- `\hat{p}_t = f_\theta(e_t, a_t, mask_t)`: dynamics model's prediction of `p_{a_t}` *before* revealing it.

### Prediction error / residual
```
r_t = p_{a_t} - \hat{p}_t                       # raw residual ∈ R^384
\hat{σ}^2_d = running variance of r_t per dim d  # online estimator
z_t = r_t / sqrt(\hat{σ}^2 + ε)                  # precision-weighted residual
```

`z_t` is what Agent C will see in its observation. For Agent B, the intrinsic reward bonus is `β * ||r_t||^2` (raw residual norm, ICM-style). The two formulations intentionally use the same dynamics model - the ablation isolates how the signal is **consumed**, not how it's computed.

### Why predict next *patch* embedding, not next *full* embedding
- Predicting the full next `e_{t+1}` couples the prediction to the agent's own update logic (patch overwrite), making the residual mostly noise from the foveation operator itself.
- Predicting the patch embedding `p_a` directly is a cleaner target: "given what I can see, what would the next patch look like if I revealed it?" Mismatch is genuinely informative about regime (e.g., heavy noise → unpredictable patches → large residual).

---

## 4. Architecture (`models/dynamics.py`)

```
input:
  low_res_emb  : (B, 384)
  action       : (B,) int in [0, 49)
  mask         : (B, 49)

embed:
  a_emb = nn.Embedding(49, 256)(action)        → (B, 256)
  ctx   = MLP([512, 256])(concat([low_res_emb, mask.float()]))   → (B, 256)
  tok0  = ctx + a_emb                          → (B, 256)

transformer:
  patch_tokens = nn.Embedding(49, 256)(arange(49))[None].expand(B, 49, 256)
  tokens = concat([tok0[:, None, :], patch_tokens], dim=1)        → (B, 50, 256)
  tokens = TransformerEncoder(layers=2, heads=4, d=256)(tokens)
  selected = tokens[batch_idx, 1 + action]                        → (B, 256)

head:
  pred_patch_emb = Linear(256 → 384)(selected)                    → (B, 384)
```

Parameter count: ~3M. Small enough to train in <1 hour on offline rollouts.

**Test (`test_dynamics.py`):** shape checks, that the same (e_t, a_t, mask_t) is deterministic, and that the model can overfit a 1k-sample subset to near-zero loss (sanity for capacity).

---

## 5. Precision estimator (`algos/precision.py`)

Welford's online algorithm for per-dim mean and variance of `r_t`, updated during dynamics training:

```python
class RunningStats:
    def __init__(self, dim, eps=1e-6): ...
    def update(self, x):     # x: (B, dim)
        # batched Welford update
    def normalize(self, x):  # x: (B, dim) → precision-weighted
        return x / (self.std + self.eps)
```

Statistics are saved into the dynamics-model checkpoint so eval reproduces exactly.

**Test (`test_precision.py`):** running stats match `np.mean/np.std` on a held-out sample within 1e-5.

---

## 6. Rollout collection (`experiments/collect_rollouts.py`)

To train the dynamics model, we need (e_t, a_t, mask_t, p_{a_t}) tuples representative of what Agent C will see in deployment. Sources:

- **Random policy** (cheapest, broad action coverage).
- **Partially-trained Agent A** (more realistic action distribution).

v1 plan: collect **500k transitions, half from random and half from a 50%-trained Agent A checkpoint**. Mix is balanced across corruption types from the training-corruption set + clean (so the dynamics model sees the full distribution it'll be evaluated on at policy-train time).

`collect_rollouts.py` runs on Modal, dumps a single `.npz` or `.pt` file per source per corruption to `/data/dynamics_rollouts/`. Total dataset size: ~1.5GB.

---

## 7. Dynamics training loop (`algos/dynamics_train.py`)

```
loss = MSE(\hat{p}_t, p_{a_t})
optimizer: Adam(lr=3e-4, wd=1e-5)
batch_size = 256
epochs = 10
```

During training, also update `RunningStats` on the residuals to feed Agent C later.

Train/val split: 90/10. Track per-corruption MSE in addition to overall - we want to see that the model is learning to predict on every corruption (no collapse).

### Critical diagnostic plot
After training, plot **histograms of `||r_t||` separated by corruption_id**. If those histograms are visually separable, the residual is a useful signal and Agent C has a chance. If they overlap heavily, the dynamics model is either too weak (residual = "irreducible task entropy") or too strong (residual = "noise"). Adjust capacity until they separate.

This plot is the actual gate to Phase 4. **If the histograms aren't separable, do not move on.**

---

## 8. Schedule: pretrain-then-freeze vs joint with PPO

**v1 commits to: pretrain then freeze.** Reasons:
- Decouples a confounder: if dynamics is updating during PPO, then Agent C's "feature distribution" is moving non-stationarily, which makes PPO harder.
- Cheaper: one dynamics training run is reused across all three seeds of Agent B and Agent C.
- Cleaner ablation: the prediction-error signal is held identical in form for B and C; only the consumption channel differs.

**Risk:** if Agent A's behavior changes a lot during PPO, the dynamics model is mildly out-of-distribution at later training stages. Mitigation: when collecting rollouts in Section 6, include partially-trained Agent A states; this widens coverage. Re-evaluate during pairing.

Alternative (joint training): plumb the dynamics optimizer into the PPO update loop, run a dynamics-loss step on each rollout batch. Defer unless Mouhssine prefers it.

---

## 9. Wandb logging during Phase 3

Per-epoch:
- `dynamics/train_loss`, `dynamics/val_loss`
- `dynamics/per_corruption_val_loss`
- `dynamics/residual_norm_mean`, `dynamics/residual_norm_std`
- Histogram: `dynamics/residual_norm_by_corruption` (one curve per corruption_id)
- `dynamics/precision_mean`, `dynamics/precision_std` (sanity for the running estimator)

---

## 10. Checkpoint format

`/ckpt/dynamics/v1.pt`:
```python
{
    "dynamics_state": ...,
    "running_stats": ...,
    "rollout_sources": ["random", "agentA_50pct_seed42"],
    "trained_corruptions": [...],
    "config": ...,
    "git_sha": ...,
}
```

This single artifact is consumed by both Agent B and Agent C.

---

## 11. Gate to Phase 4

Move on when all of these hold:

- [ ] `test_dynamics.py`, `test_precision.py` pass.
- [ ] Dynamics val loss plateaus at a non-trivial value (not zero - that means the task is too easy and residual is meaningless).
- [ ] Per-corruption residual norm histograms are **visibly separable** (the eyeball test from Section 7).
- [ ] Running precision statistics are stored in the checkpoint and reload deterministically.
- [ ] Dynamics checkpoint is on `/ckpt/dynamics/` and loadable from Modal.

---

## 12. Pairing-session checklist

Things to bring to the Week 5 session with Mouhssine:

1. The default architecture above + 1-2 alternates.
2. The diagnostic plot of residual histograms by corruption (even from a quick 10-min training run).
3. The "joint vs frozen" question with my reasoning.
4. The precision-weighting choice (per-dim running variance).
5. A simple visualization of the residual itself on a couple of canonical cases (clean image, fog, gaussian noise) to confirm the intuition matches the toy work.
