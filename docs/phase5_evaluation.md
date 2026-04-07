# Phase 5: Evaluation, Figures, and Ablations (Weeks 8-9)

Goal: produce the deliverables that answer the research question - two figures and one table, plus the written analysis. Week 8 runs the sweeps, Week 9 is buffer + figure polish.

Prerequisite: [phase4_agents_bc.md](phase4_agents_bc.md) gates green; 9 final checkpoints exist (3 agents × 3 seeds).

---

## 1. The deliverables

1. **Figure 1.** Pareto curve: accuracy vs. patch budget for Agents A, B, C on **held-out corruptions**. One panel per held-out corruption (5 panels) plus one aggregate panel.
2. **Figure 2.** Adaptation curve: accuracy as a function of number of held-out-corruption episodes the policy has streamed at test time, zero-shot. One line per agent.
3. **Table.** Final regret per agent × corruption type, averaged over 3 seeds with stddev.

The research question's one-line answer comes from comparing Agent A and Agent C in Figure 1 and Figure 2. Agent B is supporting evidence.

---

## 2. Files to create

```
src/foveated/
└── experiments/
    ├── eval_agent.py          # NEW: single-agent held-out eval
    ├── eval_sweep.py          # NEW: orchestrates all 9 evals on Modal
    ├── make_figures.py        # NEW: produces fig1, fig2, table from CSVs
    └── make_table.py          # NEW: regret table
configs/
└── eval.yaml
results/                       # gitignored; raw eval CSVs land here
deliverables/
└── milestone/
    └── figures/               # final PDFs/PNGs
```

---

## 3. Evaluation protocol (locked here - do not change after starting)

### 3.1 Held-out conditions

- **Held-out corruptions:** brightness, contrast, elastic_transform, pixelate, jpeg_compression (5 total, from [phase1_environment.md](phase1_environment.md)).
- **Severities evaluated:** 1, 3, 5 (low / medium / high; severity 5 is also out-of-training).
- **Per (corruption, severity) cell:** 1000 evaluation episodes, fresh image samples (no overlap with any training-time image even on different corruption types).
- **Goal vector at eval:** zero vector (the "unknown" embedding the policy was trained against via dropout).

### 3.2 Figure 1 - Pareto curve

We need accuracy at varying patch budgets. Two ways to get this:

**Option A - train-time sweep:** retrain each agent at multiple values of `patch_cost ∈ {0.01, 0.02, 0.05, 0.1}`. Each gives a different equilibrium patch count. 4× the training compute - too expensive.

**Option B - eval-time budget cap (chosen):** at eval, allow the agent to take its trained actions but cap reveals at `K ∈ {0, 2, 4, 6, 8}`. When the cap is hit, force `step` to produce a no-op (mask all actions). Compute accuracy at each cap.

Each agent therefore produces a curve of `(mean_accuracy, K)` per (corruption, severity). Plot aggregate over corruptions in the headline panel; per-corruption breakdown in supplementary.

**Implementation note:** evaluate the K=8 case as the agent's natural behavior (no cap). For K<8, force-truncate after K reveals; classify on whatever is revealed.

### 3.3 Figure 2 - Adaptation curve

The "adaptation" axis is **number of held-out-corruption episodes streamed**, zero-shot. The policy is **not** updated during eval. The curve shows how accuracy looks as you average over an increasing prefix of episodes - equivalent to plotting cumulative mean accuracy.

This is mostly a stability/variance story: if Agent C has higher cumulative accuracy earlier, it adapts faster (in the sense that its zero-shot behavior at first-encounter is already good).

For a stronger version of "adaptation," we'd need test-time gradient updates, which is deferred. v1 commits to zero-shot streaming.

### 3.4 Regret table

For each agent × corruption type:
- Compute "optimal" extrinsic return as the return from revealing all 49 patches, then classifying (oracle ceiling, ignores patch cost).
- Compute "agent" extrinsic return at the trained equilibrium (no cap).
- Regret = optimal − agent. Lower is better.
- Average over 3 seeds, report ± std.

---

## 4. `eval_agent.py` interface

```bash
modal run experiments/eval_agent.py \
  --ckpt /ckpt/agentC/seed42/best.pt \
  --corruptions held_out \
  --severities 1,3,5 \
  --budgets 0,2,4,6,8 \
  --n_episodes_per_cell 1000 \
  --out /results/agentC_seed42.csv
```

CSV schema (one row per episode):
```
agent, seed, corruption, severity, budget_cap,
n_revealed, predicted_class, true_class, correct, terminal_reward, episode_idx
```

This long-form CSV is the single source of truth for both figures and the table. Pandas does the rest.

---

## 5. `eval_sweep.py`

Orchestrates 9 (agent × seed) × 5 (corruption) × 3 (severity) × 5 (budget) = ~675 eval cells, but most are batched inside one Modal call per (agent, seed). Total Modal jobs: 9.

Each job takes ~30 min on A10G. Total: ~5 GPU-hours.

Resume-safe: skip jobs whose output CSV already exists.

---

## 6. `make_figures.py` and `make_table.py`

Pure pandas + matplotlib, runs locally:

```bash
python -m foveated.experiments.make_figures --results-dir results/ --out deliverables/milestone/figures/
python -m foveated.experiments.make_table   --results-dir results/ --out deliverables/milestone/regret_table.tex
```

Style notes:
- Use the same 3-color palette for A/B/C across all figures.
- Shade ±1 std (across seeds) on every curve.
- Show 3 individual seed traces lightly behind the mean for transparency.
- Captions are written into the LaTeX deliverable, not the figure file.

---

## 7. Statistical reporting

For each headline comparison (A vs. C aggregate accuracy on held-out, Agent B vs. A, Agent B vs. C):
- Bootstrap 95% CI across (seed × episode) pairs.
- Paired comparison (same image sample seen by each agent at eval - use the same `episode_idx` seed) so the comparison is paired-bootstrap, not independent.
- Report effect size (mean diff), CI, and number of paired comparisons.

Avoid p-values - they invite over-claiming. CIs and effect sizes are the right register for a class project with 3 seeds.

---

## 8. Ablations (if time allows in Week 8)

In rough priority order. Each is a single Modal job, ~3 GPU-hours.

1. **Dynamics model size for Agent C.** Train smaller and larger dynamics models; retrain Agent C on each; compare. Tests Mouhssine's "dynamics model size" sweep.
2. **Full-residual feature for Agent C** instead of per-patch norms.
3. **Goal-channel ablation for Agent A.** Drop goal entirely from Agent A's obs; does it hurt or help on held-out? Answers whether the goal channel itself does any work.
4. **Intrinsic weight robustness** for Agent B (re-run β = 0.5× and 2× the picked value).
5. **Severity generalization curve.** Eval at severities 1-5 (not just 1, 3, 5), check whether C's advantage grows with severity.

Week 9 is buffer; do not start new ablations in Week 9, just finish runs and write up.

---

## 9. Possible outcomes and how to report each

- **Agent C > A on held-out.** Headline: "prediction error as feature yields faster zero-shot adaptation on ImageNet-C." Strong positive; feeds directly into the Arbabian Lab adaptive-sensing project.
- **Agent B > A but Agent C ≈ A.** "Prediction error works as intrinsic reward but not as observation feature; sufficient-statistic claim doesn't transfer to vision at this scale." Clean negative for the lab claim, positive for ICM.
- **Agent A ≈ B ≈ C.** "Prediction error didn't matter at this scale on ImageNet-C." Useful regime characterization, a real result; do not be tempted to manufacture an effect.
- **Agent A > C.** Unexpected. Most likely cause: pred_err feature destabilizes the encoder. Report honestly with the destabilization analysis.

The writeup should be drafted **before** all the results are in (in Week 8), with the analysis sections parameterized so plugging in any of these outcomes produces a coherent paper.

---

## 10. Gate to Phase 6 (writeup)

Move on when:

- [ ] All 9 eval CSVs produced and committed under `results/`.
- [ ] Figures 1 and 2 generated, look reasonable.
- [ ] Regret table generated.
- [ ] Statistical bootstrap script produces CIs without errors.
- [ ] At least one ablation completed (dynamics model size strongly preferred).

---

## 11. What is **not** included in v1 (explicit non-goals)

- Online test-time adaptation (gradient updates at eval). Punted in Q5.
- Meta-RL extension. Mouhssine listed as Week 8-only stretch; defaulting to skip.
- Multi-corruption-per-episode (mixing corruption mid-episode). Out of scope.
- Comparison against non-foveated baselines (full-image classification). The relative claim is between A/B/C, not against a fixed baseline.
