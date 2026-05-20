# Reproduce

Full PPO + ImageNet training is a 100+ GPU-hour run that lives in
`infra/modal_app.py::train`. The decision-layer pipeline that
produces every artefact under `final_project/results/` is fully
reproducible from a clean Modal account in under a minute on CPU:

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install modal numpy matplotlib
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
cd final_project
modal run infra/modal_results.py::main
```

Default: 8 seeds (0..7), n_episodes=400, K in `{4,6,8,10,12,16}`. The
per-cell RNG is `blake2b("pareto"|seed|K|agent|corruption|episode)`,
so re-runs are byte-identical given the same seed list.

Headline checks after a run:

* `final_project/results/pareto.csv` has 30 rows (5 agents x 6 Ks);
  K=8 row for agent C should land near 0.872 +/- 0.001.
* `final_project/results/significance.json` reports paired permutation
  tests; `C_vs_A_held_out.p_two_sided` should be < 1e-3.
* `final_project/results/regression.json` reports the exponential
  approach-rate fit; `alpha` for agent C should be ~0.10 with R^2 >
  0.99.
* `final_project/results/severity.csv` and `training.csv` provide the
  noise-robustness and learning-curve breakdowns.
