# Phase 2: PPO + Agent A Baseline (Weeks 3-4)

Goal: a working goal-conditioned PPO agent (Agent A) that learns to solve the foveated MDP on clean ImageNet first, then on the 10 training corruptions of ImageNet-C. The PPO implementation here is the **shared codebase for all three agents** - Agents B and C in Phase 4 only change the observation / reward, not the optimizer.

Prerequisite: [phase1_environment.md](phase1_environment.md) gates green.
Next: [phase3_dynamics_model.md](phase3_dynamics_model.md).

---

## 1. Decisions locked here

- **Algorithm:** PPO-clip with GAE. Single-file implementation, cleanrl-style, but adapted for dict observations and vectorized GPU envs.
- **Goal conditioning:** one-hot of 16 (15 corruptions + clean) concatenated into the policy/value input. Goal is provided at **training time only**; at evaluation on held-out corruptions, the goal channel is replaced by a learned "unknown" embedding (see Section 5).
- **Architecture:** shared encoder over the dict obs; separate small heads for policy logits and value.
- **No recurrent core in v1.** The 8-step horizon is short enough that the mask carries enough history. Revisit only if Agent A fails to learn.
- **Invalid action handling:** mask invalid actions in the policy logits (set logits for revealed patches to `-inf`). Cleaner than relying on the env's "no-op penalty," and standard for action-masked PPO.

---

## 2. Files to create

```
src/foveated/
├── models/
│   └── policy.py
├── algos/
│   ├── __init__.py
│   ├── ppo.py
│   ├── rollout_buffer.py
│   └── ppo_loss.py
├── experiments/
│   ├── __init__.py
│   └── train_agent.py
└── utils/
    ├── schedulers.py
    └── checkpoint.py
configs/
├── base.yaml
└── agent_a.yaml
tests/
├── test_policy.py
├── test_rollout_buffer.py
└── test_ppo_step.py
```

---

## 3. Policy / value architecture (`models/policy.py`)

```
input: dict obs
  low_res_emb : (B, 384)
  mask        : (B, 49)
  goal        : (B, 16)
  pred_err    : (B, D_ERR)         # zeros for Agent A

encoder:
  x = concat([low_res_emb, mask.float(), goal, pred_err])  → (B, 384 + 49 + 16 + D_ERR)
  h = MLP([256, 256], activation=Tanh)(x)                  → (B, 256)

policy head:
  logits = Linear(256 → 49)(h)
  logits = logits.masked_fill(mask == 1, -inf)             # action masking

value head:
  v = Linear(256 → 1)(h)
```

Notes:
- `Tanh` not `ReLU`. PPO with `Tanh` is the historical default and tends to be more stable on small networks.
- Orthogonal init with `gain=sqrt(2)` for hidden layers, `gain=0.01` for the policy head, `gain=1.0` for the value head. Standard PPO init.
- The encoder is shared across all three agents. Agent C will simply have `D_ERR > 0`; the rest is identical. **Do not introduce per-agent architecture changes** - that would confound the ablation.

**Test (`test_policy.py`):** forward pass shapes, gradient flows to the encoder, masked actions get `-inf` logits, sampling never produces a masked action.

---

## 4. Rollout buffer (`algos/rollout_buffer.py`)

Standard PPO buffer, but:
- Stores dict observations as a dict of tensors (no flattening).
- All on GPU. The vectorized env returns GPU tensors; the buffer never moves to CPU.
- Capacity = `n_envs * n_steps`. After collection, GAE is computed batched on GPU.

GAE parameters: `gamma=0.99`, `lambda=0.95`. Because the episode is only 8 steps and reward is terminal-only, gamma matters a lot - try `gamma=1.0` as an early ablation if learning is slow.

**Test (`test_rollout_buffer.py`):** advantages with known returns match a numpy reference implementation.

---

## 5. Goal conditioning under shift

At train time the goal channel is one-hot. At eval on held-out corruptions we have several options:

1. Zero vector for unknown goals.
2. A dedicated 17th index for "unknown."
3. A learned "unknown" embedding trained by random goal-dropout during training.

**v1 commits to option 3 (dropout-trained unknown):** with probability `p_dropout=0.1`, replace the goal one-hot with the zero vector during training. At eval on held-out corruptions, the goal is the zero vector. This gives the policy a calibrated behavior for "I don't know the regime," which is precisely the setting Agent C is supposed to handle by inferring from prediction error.

If the goal should be present at eval too, this is the spot to change. See Q1 in [design_decisions.md](design_decisions.md).

---

## 6. PPO loop (`algos/ppo.py`)

Standard PPO-clip:

```
for update in range(num_updates):
    obs, actions, log_probs_old, returns, advantages = collect_rollout(env, policy, n_steps)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for epoch in range(ppo_epochs):
        for mb in minibatches(...):
            logits, value = policy(mb.obs)
            log_probs = log_softmax(logits).gather(mb.actions)
            ratio = exp(log_probs - mb.log_probs_old)
            policy_loss = -min(ratio * mb.adv, clip(ratio, 1-ε, 1+ε) * mb.adv).mean()
            value_loss = ((value - mb.returns) ** 2).mean()
            entropy = -(softmax(logits) * log_softmax(logits)).sum(-1).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optim.step(loss)
    log to wandb; checkpoint every K updates
```

### Hyperparameters (locked across A/B/C; `configs/base.yaml`)

```yaml
algo:
  name: ppo
  total_env_steps: 5_000_000        # ~625k episodes at 8 steps each
  n_envs: 64
  n_steps: 16                       # 64*16 = 1024 transitions per update
  ppo_epochs: 4
  minibatch_size: 256
  gamma: 0.99
  lambda: 0.95
  clip_eps: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  max_grad_norm: 0.5
  lr: 3.0e-4
  lr_schedule: linear_decay         # 3e-4 → 0 over total_env_steps
  goal_dropout: 0.1

env:
  grid: 7
  horizon: 8
  patch_cost: 0.02

eval:
  n_episodes: 256
  every_updates: 25
```

Agent-A-specific config (`configs/agent_a.yaml`) only sets `pred_err_dim: 0` and the wandb tags.

### Compute budget estimate
`5e6` env steps × 1 ViT forward per step ≈ 5e6 forwards. ViT-small at batch 64 on an A10G runs ~500 batches/s, so ~5e6 / 64 / 500 ≈ 160s of compute on env forwards, plus PPO updates. Realistic wall-clock on A10G: 3-6 hours per seed including overhead. Budget 3 seeds × Agent A = ~15 GPU-hours.

---

## 7. Training schedule

### Step 2.1 - Clean-only sanity training
Train Agent A on **clean ImageNet only** (no corruptions, no goal channel needed - pass zeros). Target: monotonic improvement in terminal reward; final accuracy near "all-49-patches" head accuracy at the chosen patch cost.

This is a debugging milestone, not part of the final results. If clean-only learning doesn't work, debug here before adding goal conditioning.

### Step 2.2 - Goal-conditioned on training corruptions
Switch to ImageNet-C with the 10 training corruptions, severities 1-3 (`splits.py`). Goal one-hot, dropout 0.1. Same hyperparameters. Target: training-distribution accuracy comparable to clean-only baseline within ~5pp.

### Step 2.3 - 3-seed run
Once a single seed works, launch 3 seeds (e.g., 42, 1337, 2024) and capture learning curves. This produces the Agent A reference for the final report.

---

## 8. Wandb logging during Phase 2

Per-update metrics:
- `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/approx_kl`, `train/clip_fraction`
- `train/lr`, `train/grad_norm`
- `train/explained_variance`

Per-evaluation metrics (every 25 updates, on 256 held-out training-distribution episodes):
- `eval/accuracy`
- `eval/mean_reward`
- `eval/mean_patches_revealed`
- `eval/per_corruption_accuracy` (a small bar chart)

Sanity flags (auto-alert):
- `approx_kl > 0.05` for >5 updates → step size too high.
- `clip_fraction > 0.4` → same.
- `explained_variance < 0` → value function broken.

---

## 9. Checkpoint format

`/ckpt/agentA/seed{N}/update_{K}.pt`:
```python
{
    "policy_state": ...,
    "optim_state": ...,
    "running_obs_stats": ...,   # if we normalize obs
    "config": ...,              # full config dict
    "global_step": ...,
    "git_sha": ...,
}
```

Keep the most-recent and best-eval checkpoint per seed. Delete intermediate ones; data is cheap but listing 200 ckpts per agent is annoying.

---

## 10. Gate to Phase 3

Move on when all of these hold:

- [ ] `test_policy.py`, `test_rollout_buffer.py`, `test_ppo_step.py` pass.
- [ ] Single-seed clean-only Agent A learns (accuracy curve climbs and plateaus above random-policy baseline by >10pp).
- [ ] Single-seed goal-conditioned Agent A learns on 10 training corruptions; final eval accuracy on the training-corruption set is within 5pp of clean-only.
- [ ] 3-seed Agent A run is launched on Modal. Phase 3 can begin while these run.
- [ ] At least one Agent A checkpoint is saved and reloadable.

---

## 11. Open questions deferred

- **Q1 (goal at eval).** v1 uses dropout-trained "unknown" goal at eval on held-out corruptions. Revisit before Step 2.2.
- **Q5 (adaptation curve semantics).** Confirmed as zero-shot streaming for the final eval. No change needed in Phase 2 - affects Phase 5 only.
- **Q10 (compute budget).** Estimate above is ~15 GPU-hours for Agent A alone (3 seeds). Total project budget across A+B+C and ablations is ~80-120 GPU-hours. Revisit if this is too aggressive.

If clean-only learning in Step 2.1 fails to converge: do **not** start adding tricks. The first thing to try is reducing `patch_cost` (so positive reward is easier) and increasing entropy bonus (so exploration broadens). If those don't help, the env or the head is at fault - go back to Phase 1.
