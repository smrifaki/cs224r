# Foveated adaptive sensing with prediction-error policy features

CS 224R project, Stanford, spring 2026.

Research question: does exposing a prediction-error signal as a
policy observation feature give a deep RL agent faster adaptation
under distribution shift, compared to using it as an intrinsic
reward bonus or omitting it?

**Real ImageNet-100 PPO sweep on Modal T4** (4 agents × 3 seeds,
60k env-steps each, frozen pretrained ViT-small backbone, K=8 of
49 patches):

| agent | mechanism                          | mean eval acc |
|-------|------------------------------------|--------------:|
| A     | baseline                           | 0.845         |
| B     | intrinsic reward = β · ‖residual‖² | 0.823         |
| C     | precision-weighted residual obs    | 0.833         |
| D     | assembly entropy obs               | 0.836         |

Backbone whole-image val acc: 0.882-0.901. Random baseline: 0.010.
The K=8 PPO agents recover ~94% of the backbone ceiling. Per-seed
breakdown + dynamics-pretrain log in
[results/real_imagenet/](results/real_imagenet/).

The C > A claim depends on distribution shift; the next sweep adds
ImageNet-C corruptions and the held-out evaluation. Code for that
is the same `infra/modal_real_full.py` with a config flag.

Setting: foveated image classification on ImageNet (currently a
100-class subset) and ImageNet-C (next). Four goal-conditioned PPO
agents share env, backbone, and hyperparameters and differ only in
how they consume a frozen forward-dynamics model's residual signal.

| Agent | Prediction-error channel |
|-------|--------------------------|
| A | none, baseline |
| B | intrinsic reward bonus, Pathak-style ICM |
| C | per-action prospective uncertainty as an observation feature |
| D | classifier entropy as an observation feature |

Companion course-submission repo at
https://github.com/BYTurnips/CS224R (Brion's enrolled-student copy).

## Setup

```bash
conda env create -f environment.yml
conda activate cs224r
```

CPU-only: edit `environment.yml` and change the PyTorch index URL
from `cu121` to `cpu`.

Auth as needed:

```bash
huggingface-cli login
wandb login
modal token new
```

## Running the real proposal sweep

```bash
modal run infra/modal_real_full.py::main \
  --agents A,B,C,D --seeds 0,1,2 --n-env-steps 60000 --n-classes 100
```

This launches one forward-dynamics pretrain job on A10G followed by
12 PPO training jobs in parallel on T4 (4 agents × 3 seeds). Each
job streams to wandb. Results land in `results/real_imagenet/` and
checkpoints on the `cs224r-ckpts` Modal Volume.

Full reproduction recipe in [REPRODUCE.md](REPRODUCE.md).

## Layout

```
.
  environment.yml         conda env
  pyproject.toml          package metadata + entry points
  configs/                YAML configs per agent
  src/foveated/
    data/                 ImageNet + ImageNet-C wrappers
    envs/                 FoveatedEnv (gymnasium), wrappers, classifier head
    models/               ViT-small backbone, dynamics model (plain + Bayesian + IB)
    algos/                PPO, rollout buffer, intrinsic reward, dynamics training
    experiments/          train_agent, evaluate, aggregate, calibration, etc.
  infra/                  Modal entry points (real_full, real_imagenet_v2, real_imagenet, app, download_data, modal_config)
  tests/                  pytest unit and integration tests
  scripts/                local orchestration
  docs/                   phase plans and design decisions
  results/real_imagenet/  real Modal-T4 PPO numbers from the sweep
  results/real_ppo/       28-s CIFAR-10 PPO smoke (validates the stack)
```

Course-submission deliverables (proposal, milestone, poster, final
paper) live in the companion repo
https://github.com/BYTurnips/CS224R, not here.

## Phase docs

| Doc | Contents |
|-----|----------|
| [docs/setup.md](docs/setup.md) | conda, PyTorch, HF, Modal, wandb |
| [docs/phase1_environment.md](docs/phase1_environment.md) | FoveatedEnv, ViT-small backbone, ImageNet-C wrappers |
| [docs/phase2_ppo_baseline.md](docs/phase2_ppo_baseline.md) | goal-conditioned PPO, Agent A three-seed run |
| [docs/phase3_dynamics_model.md](docs/phase3_dynamics_model.md) | forward dynamics, precision-weighted residual |
| [docs/phase4_agents_bc.md](docs/phase4_agents_bc.md) | Agents B (intrinsic reward) and C (obs feature) |
| [docs/phase5_evaluation.md](docs/phase5_evaluation.md) | held-out eval, Pareto curve, adaptation curve, regret table |
| [docs/design_decisions.md](docs/design_decisions.md) | implementation choices and trade-offs |
| [docs/decisions.md](docs/decisions.md) | recent design choices log |

## Deliverables

* Figure 1: Pareto curve, accuracy vs patch budget across A/B/C/D
  on held-out ImageNet-C corruptions. Requires the corruption-eval
  pass.
* Figure 2: adaptation curve, accuracy vs number of held-out-
  corruption episodes streamed.
* Table: per-agent, per-corruption regret (mean +/- std over three
  seeds).

## License

MIT. See `LICENSE`.
