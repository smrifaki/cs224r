# Decision-layer results, 2026-05-20

8 seeds, 400 episodes per cell, blake2b-derived per-cell RNG, CPU.

```bash
modal run infra/modal_results.py::main
```

## K=8 held-out top-K coverage

| agent | mean | stderr | gap vs A |
|-------|-----:|-------:|---------:|
| A (baseline)         | 0.162 | 0.001 |    -- |
| B (intrinsic reward) | 0.390 | 0.001 | +0.228 |
| C (residual feature) | 0.872 | 0.001 | +0.710 |
| D (entropy feature)  | 0.445 | 0.001 | +0.283 |
| oracle               | 1.000 | 0.000 | +0.838 |

Source: [results/pareto.csv](results/pareto.csv).

## Pareto across K

| K  | A     | B     | C     | D     | oracle |
|---:|------:|------:|------:|------:|-------:|
|  4 | 0.082 | 0.239 | 0.784 | 0.276 | 1.000  |
|  6 | 0.123 | 0.318 | 0.836 | 0.370 | 1.000  |
|  8 | 0.162 | 0.390 | 0.872 | 0.445 | 1.000  |
| 10 | 0.205 | 0.449 | 0.897 | 0.511 | 1.000  |
| 12 | 0.246 | 0.500 | 0.919 | 0.567 | 1.000  |
| 16 | 0.328 | 0.587 | 0.934 | 0.654 | 1.000  |

## Approach to oracle

Fitting `log(1 - coverage(K)) = -alpha * K + intercept` recovers the
exponential rate at which each agent approaches the ceiling.

| agent | alpha | R^2 |
|-------|------:|----:|
| A | 0.026 | 0.997 |
| B | 0.051 | 0.999 |
| C | 0.101 | 0.997 |
| D | 0.061 | 0.998 |

Agent C's approach rate is ~4x baseline. Source:
[results/regression.json](results/regression.json),
[results/figures/regression.pdf](results/figures/regression.pdf).

## Paired permutation test

Per-corruption regret, paired by (seed, corruption) on the five
held-out corruptions. 8 seeds x 5 corruptions = 40 pairs.

| comparison | mean diff in regret | p (two-sided) |
|------------|--------------------:|--------------:|
| C vs A     | -0.707 | < 1e-3 |
| C vs D     | -0.424 | < 1e-3 |

Source: [results/significance.json](results/significance.json).

## Adaptation curve

Top-K coverage as a function of episodes streamed on the held-out
corruption (K = 8, no test-time gradient updates). Agent C plateaus
above 0.85 within ~5 episodes; A stays at the random-baseline level.

Figures: [results/figures/adaptation.pdf](results/figures/adaptation.pdf),
[results/figures/pareto.pdf](results/figures/pareto.pdf),
[results/figures/regret_heatmap.pdf](results/figures/regret_heatmap.pdf),
[results/figures/composite.pdf](results/figures/composite.pdf).

## Reproducibility

The per-cell RNG is `blake2b("pareto"|seed|K|agent|corruption|ep)`,
so a re-run with the same seed list returns byte-identical CSVs.
The Modal app re-imports cleanly without GPU.

## What this is not

This is the *decision-layer* test: oracle informativeness is given,
the agents see only the feature that distinguishes them. The full
PPO + ImageNet experiment lives in `infra/modal_app.py::train` and
runs on a GPU. The decision-layer numbers reproduce the structure
the full run is expected to produce but at a fraction of the cost.
