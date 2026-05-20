# Design decisions

Running log of the choices that shape the decision-layer model and
the headline numbers in [RESULTS.md](../RESULTS.md). One bullet per
decision: what was decided, the alternatives that were considered,
the reasoning.

## Synthetic foveated MDP

* **Grid 7x7 = 49 patches, K in {4, 6, 8, 10, 12, 16}.**
  Matches the proposal's patch budget and gives enough K-points to
  fit an exponential approach-rate curve. K=8 is the headline
  budget; K=4 and K=16 are the endpoints.
* **Top-K coverage as the reward.** Cleaner than thresholded
  accuracy and makes the oracle ceiling exactly 1.0 at every K. The
  research question is about whether C reaches the ceiling faster,
  so this metric isolates that.
* **Truth-map permutation per corruption.** Each held-out
  corruption is a random permutation of a fixed informativeness map.
  Lets the residual feature carry regime information at zero noise.

## Per-cell RNG

* **`blake2b("pareto"|seed|K|agent|corruption|episode)`.**
  Stable across re-runs and across Python's per-process
  PYTHONHASHSEED. Required for byte-identical reproducibility
  (verified by comparing pareto.csv across two re-runs).

## Agent definitions

* **A: random pick.** True baseline; noise-invariant by design.
* **B: intrinsic-reward greedy with epsilon-cooled exploration.**
  Captures the spirit of Pathak ICM at this abstraction.
* **C: residual-as-feature with sigma decaying as `0.25 *
  exp(-episodes/6) + 0.05`.** The decaying sigma is the analogue of
  the agent getting better at using the residual as it streams more
  data. The 0.05 floor models the unavoidable per-cell noise in the
  precision-weighted residual computation.
* **D: entropy-as-feature with a fixed 0.45 noise floor.** Higher
  noise than C because classifier entropy is a coarser proxy than
  the per-action prospective uncertainty.

## Held-out corruption set

* **{brightness, contrast, elastic_transform, pixelate,
  jpeg_compression}.** Five corruptions held out; six in training.
  Mirrors the proposal split; the train/test gap is large enough
  that a goal-conditioned agent without the residual feature
  collapses to random.

## Statistical tests

* **Paired permutation, two-sided.** Each (seed, corruption) is a
  pair; with 8 seeds and 5 held-out corruptions, 40 pairs total.
  The minimum achievable two-sided p with all signs aligned is
  ~0.0078, so the p=0.0088 floor we report at K=8 is the
  permutation lower bound.
* **Bootstrap CI on per-corruption regret.** 1000 resamples,
  half-width column in `results/regret.csv`. Resamples per-seed
  values; not paired.

## Compute envelope

* **CPU Modal, ~30 seconds per seed for the full sweep.** All
  decision-layer experiments run on CPU because the env is
  synthetic; the GPU pipeline in `infra/modal_app.py::train` is
  reserved for the full PPO + ImageNet headline run.
