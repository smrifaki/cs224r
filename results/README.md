# results/ directory map

Real Modal-run artifacts only. Synthetic decision-layer outputs
(pareto.csv, severity.csv, etc.) were removed; their narrative was
not backed by real PPO training, just a synthetic-MDP simulation.

```
results/
  real_imagenet/
    sweep_full_A_B_C_D_seeds_0_1_2.json  full 12-job table
    sweep_v2_A_D_seeds_0_1_2.json        the v2 A+D run that fixed
                                         the feature-pooling bug
    sweep_A_D_seeds_0_1_2.json           the v1 A+D run (kept for
                                         the bug post-mortem)
    full_run_log.md                      headline table + how to run
    run_log.md                           v1/v2 comparison
  real_ppo/
    smoke_seed0.json                     28-s CIFAR-10 PPO smoke
    README.md                            honest gap analysis
```

Reproduce the full sweep:

```bash
modal run infra/modal_real_full.py::main \
  --agents A,B,C,D --seeds 0,1,2 --n-env-steps 60000 --n-classes 100
```
