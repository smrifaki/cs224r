# Phase 4: Agents B (Intrinsic Reward) and C (Observation Feature) - Weeks 6-7

Goal: train Agents B and C using the dynamics model from Phase 3, with **everything except the prediction-error consumption channel held identical to Agent A**. This is the actual ablation that answers the research question.

Prerequisite: [phase3_dynamics_model.md](phase3_dynamics_model.md) gates green; dynamics checkpoint frozen at `/ckpt/dynamics/v1.pt`.
Next: [phase5_evaluation.md](phase5_evaluation.md).

---

## 1. The cardinal rule of Phase 4

**Hold everything constant across A/B/C except the one channel being studied.** That means:

| component | A | B | C |
|---|---|---|---|
| Backbone (frozen ViT-small) | same | same | same |
| Classifier head | same | same | same |
| Env (grid, horizon, patch_cost) | same | same | same |
| PPO hyperparameters | same | same | same |
| Seeds | {42, 1337, 2024} | {42, 1337, 2024} | {42, 1337, 2024} |
| Goal conditioning + dropout | same | same | same |
| Dynamics model (frozen) | - | v1 | v1 |
| **Intrinsic reward** | none | `β * ‖r_t‖²` | none |
| **Observation** | `[low_res_emb, mask, goal, 0_384]` | `[low_res_emb, mask, goal, 0_384]` | `[low_res_emb, mask, goal, z_t]` |

If you find yourself wanting to tweak something for B or C that's not in the bolded rows, **don't** - note it in `docs/design_decisions.md` and revisit at the end.

---

## 2. Files to create / modify

```
src/foveated/
├── envs/
│   ├── foveated_env.py            # MODIFY: implement pred_err path
│   └── intrinsic_wrapper.py        # NEW: env wrapper for Agent B
├── algos/
│   └── intrinsic_reward.py         # NEW: ICM-style bonus computation
└── experiments/
    └── train_agent.py              # MODIFY: branch on agent type
configs/
├── agent_b.yaml
└── agent_c.yaml
tests/
├── test_intrinsic_reward.py
└── test_agent_c_obs.py
```

---

## 3. Wire the dynamics model into the env

`FoveatedEnv` already has a `pred_err` slot in the observation (Phase 1 reserved it). Now we populate it.

### Agent C path
- On reset and on each step, after computing `low_res_emb`, also compute predicted patch embeddings for **all unrevealed patches**, take the residual against the actual patch embeddings (extracted from `image_full`), normalize via `RunningStats`, and store.
- The full residual vector is `(49, 384) = 18816` dims - too large to feed directly. v1 strategy: feed back the **per-patch residual norm** (a 49-dim vector). This is a compact summary that says "how surprising would each unrevealed patch be."
  - Set `D_ERR = 49` in this case, not 384.
  - For revealed patches, set the entry to 0.
- This is a deliberate simplification of the proposal's "per-patch precision-weighted residual." Full residual is the natural next ablation if v1 doesn't show signal.

### Agent A and B path
- `pred_err` slot stays as zeros of size `D_ERR=49` (to keep the obs shape constant across A/B/C - same encoder).

**Test (`test_agent_c_obs.py`):** for Agent C, `pred_err` entries for revealed patches are zero; entries for unrevealed patches match what a manual computation produces.

### Performance note
Predicting 49 patches per step × 8 steps × N envs is heavier than Agent A. Profile early - if it's too slow, batch the dynamics forward across all (env, patch) pairs into one call.

---

## 4. Intrinsic reward wrapper (`envs/intrinsic_wrapper.py`)

A thin wrapper over `FoveatedEnv` used only for Agent B:

```python
class IntrinsicRewardWrapper:
    def __init__(self, env, dynamics, beta, running_stats):
        ...
    def step(self, action):
        obs, ext_reward, terminated, truncated, info = env.step(action)
        # compute residual norm for the action just taken
        r_norm = compute_residual_norm(prev_obs, action, dynamics, running_stats)
        intrinsic = beta * r_norm ** 2
        reward = ext_reward + intrinsic
        info["ext_reward"] = ext_reward
        info["intrinsic_reward"] = intrinsic
        return obs, reward, terminated, truncated, info
```

Key choices:
- Intrinsic is computed on the **patch just revealed**, using the same dynamics model. This matches ICM's "surprise of what just happened" semantics.
- `r_norm` uses raw residual norm (not precision-normalized) - that matches Pathak et al. and keeps Agent B's signal definition canonical.
- Always log `ext_reward` and `intrinsic_reward` separately so we can see if Agent B is learning extrinsic or just chasing surprise.

**Test (`test_intrinsic_reward.py`):** intrinsic reward is non-negative; with `beta=0`, total reward equals env's extrinsic reward exactly.

---

## 5. Tuning the intrinsic reward weight β

This is the one hyperparameter that has to be tuned per-agent (B only). Strategy:

1. **Small sweep on one corruption.** Run Agent B with `β ∈ {0.0, 0.01, 0.1, 1.0}` on a single training corruption (say, gaussian_noise) for ~1M env steps. Pick the β that gives highest extrinsic return.
2. **Sanity check.** That β should make intrinsic-to-extrinsic ratio peak around 0.1-0.3 in early training and decay as the dynamics residual shrinks (which it won't on the foveated MDP since dynamics is frozen - so probably stable). Watch for `β` so large that the policy ignores extrinsic reward entirely.
3. **Lock β** for the 3-seed run.

"Ablations on intrinsic reward weight" is one of the planned sweeps - that's this sweep. Budget: ~4 single-seed short runs × 1 GPU-hour each = 4 GPU-hours. If pace is tight, use Pathak et al.'s reported `β=0.01` and skip the sweep.

---

## 6. Training schedule

### Step 4.1 - Agent C, single seed
Train Agent C with `D_ERR=49` and the populated pred_err feature. Confirm:
- Learning curve climbs at least as fast as Agent A on training corruptions (since C has strictly more information).
- Per-corruption eval accuracy on training distribution matches A within 2pp.

If Agent C learns slower than A on the training distribution, that's a sign the extra feature is destabilizing the encoder. Tactics: lower lr, increase warmup, freeze the encoder for the first ~50k steps.

### Step 4.2 - Agent B β sweep
Run the 4-run sweep above. Pick β.

### Step 4.3 - Agent B, single seed at locked β
Same as 4.1 but for Agent B.

### Step 4.4 - 3-seed runs for B and C
Once both single-seed runs are clean, launch 3 seeds each. Total compute estimate: ~30 GPU-hours.

---

## 7. Wandb logging additions

Beyond what's in [phase2_ppo_baseline.md](phase2_ppo_baseline.md):

For Agent B:
- `train/intrinsic_reward_mean`
- `train/extrinsic_reward_mean`
- `train/intrinsic_to_extrinsic_ratio`

For Agent C:
- `obs/pred_err_norm` histogram (per minibatch)
- `obs/pred_err_per_corruption` (eval-time, separated by goal - even though policy sees zero goal at eval, the env logs ground truth)
- `obs/pred_err_zero_fraction` (sanity: should equal `mask.sum()/49`)

---

## 8. Common pitfalls and what to do about them

These are predictable failure modes; if any of them happen, take the listed step, not a creative fix.

1. **Agent C plateaus at Agent A's level on training distribution.** The pred_err feature isn't being used. Check that the encoder gradients actually flow through that slice of the input (set a breakpoint, inspect grads). If yes, the feature is just not informative - go back to Phase 3 and reconsider the residual definition.

2. **Agent C is *worse* than Agent A on training.** The feature is destabilizing. Reduce lr by 3×, or freeze the encoder weights connected to the pred_err slice for the first 100k steps (warmup).

3. **Agent B's policy collapses to revealing all patches** (intrinsic reward incentivizes revealing the most surprising patches). Lower β; or normalize intrinsic by extrinsic running mean so the ratio stays bounded.

4. **Agent B's policy collapses to revealing none.** β is so small it's noise; ext_reward gradient dominates and patch_cost makes "reveal nothing" optimal. Increase β; or check that early training has any reveals at all (entropy bonus issue).

5. **Dynamics model drifts at inference** (rare since it's frozen, but the running stats might if you accidentally update them). Make sure `RunningStats.update()` is **not called** during RL training - only during Phase 3.

6. **Pred_err feature distribution shifts on held-out corruptions** in a way that breaks Agent C. Expected - the whole point - but it shouldn't be catastrophic. If it is, the precision normalization in Phase 3 wasn't computed over a representative-enough distribution.

---

## 9. Gate to Phase 5

Move on when all of these hold:

- [ ] `test_intrinsic_reward.py`, `test_agent_c_obs.py` pass.
- [ ] Agent C, 3 seeds, completed training. Eval on training corruptions matches Agent A within 2pp.
- [ ] Agent B, 3 seeds, completed training at locked β. Eval on training corruptions matches Agent A within 2pp.
- [ ] No agent has a pathological behavior (always-reveal-all or never-reveal).
- [ ] All 9 checkpoints (3 agents × 3 seeds) are on `/ckpt/` and loadable.

---

## 10. Stretch goals for end of Phase 4 (only if ahead of schedule)

- **Full residual as feature** (`D_ERR=49*384`) instead of per-patch norms, with a learned projection down to 49 dims inside the encoder.
- **Joint dynamics training** during PPO (the alternative from [phase3_dynamics_model.md](phase3_dynamics_model.md) Section 8).
- **Stop action** (`Discrete(50)`), terminating early when the agent thinks it has enough info. This is a real research direction in its own right; pursue only with a clear hour budget.
