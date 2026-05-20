# results/ directory map

Every artefact under `results/` is produced by a single `modal run`
of `infra/modal_results.py::main`. The full reproduction recipe is
in [REPRODUCE.md](../REPRODUCE.md).

```
results/
  pareto.csv                   30 rows: 5 agents x 6 patch budgets;
                               mean coverage + stderr + n_seeds
  adaptation.csv               4 agents x 30 streamed episodes; mean
                               coverage at K=8 on held-out corruptions
  training.csv                 per-agent training-curve on training
                               corruptions, 80 episodes, K=8
  regret.csv                   per (corruption, agent) regret vs the
                               oracle, with bootstrap CI half-width
                               and held-out flag
  severity.csv                 3D ablation: agent x K x noise_scale,
                               accuracy + stderr. 3 Ks x 5 scales x 4
                               agents = 60 rows
  significance.json            paired permutation tests:
                                 C vs A held-out (5 corruptions, 8
                                 seeds = 40 pairs)
                                 C vs D held-out
                                 per-K paired test for C vs A
  regression.json              fits for log(1 - coverage(K)) per agent
  sample_efficiency.json       per-agent episodes to reach 50/75/95%
                               of final-window mean training accuracy
  figures/
    pareto.{pdf,png}
    adaptation.{pdf,png}
    training_curve.{pdf,png}
    regret_heatmap.{pdf,png}
    severity.{pdf,png}
    regression.{pdf,png}
    k8_bar.{pdf,png}
    composite.{pdf,png}
```

The decision-layer pipeline is reproducible by stable blake2b
seed derivation. The full GPU PPO + ImageNet path lives in
`infra/modal_app.py::train` and produces the same shape of
artefacts but on real data.
