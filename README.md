# Foveated adaptive sensing with prediction-error policy features

CS 224R project, Stanford, spring 2026. Authors: Mouhssine Rifaki and
Brion Ye.

Research question: does exposing a prediction-error signal as a
policy observation feature give a deep RL agent faster adaptation
under distribution shift, compared to using it as an intrinsic
reward bonus or omitting it?

Setting: foveated image classification on ImageNet and ImageNet-C.
Four goal-conditioned PPO agents share env, backbone, and
hyperparameters and differ only in how they consume a frozen
forward-dynamics model's residual signal.

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

## Running experiments

Long-running jobs go to Modal; smoke runs and figures are local.

```bash
# one-time data fetch
modal run infra/download_data.py

# smoke train
modal run infra/modal_app.py::smoke

# train one agent at one seed
modal run infra/modal_app.py::train --agent A --seed 42

# fan out
python scripts/train_all.py

# evaluate held-out corruptions
python scripts/eval_all.py

# figures and regret table
python scripts/make_all_figures.py
```

CLI entry points after install:

```bash
fov-train --agent C --seed 1337 --config configs/agent_c.yaml
fov-eval  --ckpt /ckpt/agentC/seed1337/best.pt --out results/agentC_seed1337.csv
```

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
  infra/                  Modal image, app, data-download
  tests/                  pytest unit and integration tests
  scripts/                local orchestration (smoke, train-all, figures)
  docs/                   phase plans and design decisions
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

## Deliverables

* Figure 1: Pareto curve, accuracy vs patch budget for A/B/C/D on
  held-out corruptions.
* Figure 2: adaptation curve, accuracy vs number of held-out-
  corruption episodes streamed.
* Table: per-agent, per-corruption regret (mean +/- std over three
  seeds).

## License

MIT. See `LICENSE`.
