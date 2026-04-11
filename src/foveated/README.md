# `foveated` package layout

Cross-reference for every module in the `foveated` package. Each
module has a single responsibility; the architecture is intentionally
flat and the import graph has no cycles.

## `envs/`

* **`foveated_env.py`** is the gym environment plus a `BackboneAdapter`
  that wraps a timm classifier so the env can extract low-res and
  per-patch embeddings without touching backbone internals. The
  corruption pipeline (ImageNet-C, with a Gaussian-noise fallback)
  lives here.
* **`intrinsic_wrapper.py`** is Agent B's reward wrapper. Adds a
  Pathak-style intrinsic bonus computed from the previous step's
  precision-weighted residual. Beta anneals linearly.
* **`feature_wrappers.py`** is Agents C and D. The
  `ProspectiveUncertaintyObsWrapper` exposes per-action predicted
  log-variance (or, with a Laplace posterior, the epistemic
  component). The `ClassifierEntropyObsWrapper` exposes the
  classifier's entropy of the current assembly.

## `models/`

* **`dynamics.py`** is the forward dynamics model with a Gaussian
  $(\mu, \log\sigma^2)$ head. Trained by NLL. Exposes
  `query_all_actions` and `prospective_uncertainty` for the
  decision-time signal.
* **`dynamics_bayesian.py`** is the last-layer Laplace approximation
  on top of a trained `ForwardDynamics`. Returns a
  `LaplacePosterior` that the feature wrapper can consume for the
  BALD-aligned epistemic uncertainty.
* **`dynamics_ib.py`** is the information-bottleneck variant of
  the dynamics objective. A reparameterized stochastic latent plus a
  $\beta$-weighted KL to a unit Gaussian prior. $\beta = 0$ recovers
  the plain NLL.

## `algos/`

* **`precision.py`** computes the precision-weighted residual.
* **`intrinsic_reward.py`** wraps the precision-weighted residual
  into the Pathak intrinsic bonus.
* **`dynamics_train.py`** pretrains the dynamics model on random-
  policy rollouts. CLI: `python -m foveated.algos.dynamics_train`.
* **`stats.py`** is the statistics module: bootstrap CI, paired
  permutation test, Cliff's delta, `report_pairwise` helper.

## `data/`

* **`imagenet.py`** is the manifest loader plus a thin `load_backbone`
  helper.

## `experiments/`

| Script | Purpose |
|---|---|
| `collect_rollouts.py` | random-policy rollouts producing $(z_t, a_t, z_{t+1})$ triples |
| `train_agent.py` | goal-conditioned PPO for A / B / C / D |
| `evaluate.py` | Pareto, adaptation, regret pipeline; writes the per-seed eval artifact |
| `fit_laplace.py` | post-training Laplace posterior fitting |
| `calibration.py` | reliability diagram and ECE per corruption |
| `severity_sweep.py` | top-1 vs ImageNet-C severity per agent |
| `feature_ablation.py` | inference-time ablation of Agent C's prospective slot |
| `oracle_topk.py` | greedy clairvoyant K-patch oracle |
| `policy_viz.py` | per-agent patch-commitment heatmaps |
| `aggregate_seeds.py` | pool eval results across seeds; emit pairwise stats |
| `composite_figure.py` | five-panel results figure |
| `online_adaptation.py` | test-time updates of the dynamics variance head |
| `lipschitz_estimate.py` | empirical L and M for the regret bound |
| `validate_bound.py` | fit regret-bound predictions P1 and P2 |
| `budget_sweep.py` | regret vs patch budget K |
| `per_corruption_breakout.py` | per-corruption small-multiples figure |
| `adversarial_test.py` | failure-mode probe; Spearman correlation of uncertainty vs marginal gain |

## Module dependency convention

* `envs/` may import from `models/`, `algos/`, but not from `experiments/`.
* `models/` may import from each other and from `algos/`. No env or data.
* `algos/` is leaf; no project imports except other `algos/` modules.
* `data/` may import from `envs/` for the `BackboneAdapter` type.
* `experiments/` is top of the stack; may import anything.

No circular imports. CI lint enforces.
