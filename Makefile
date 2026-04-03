.PHONY: all manifest dynamics laplace train eval aggregate composite test lint clean

PY := python
SEEDS := 42 1337 2024
AGENTS := A B C D
IMAGENET_VAL ?= /data/imagenet/val
MANIFEST := manifest.json
SMOKE_MANIFEST := smoke_manifest.json
DYNAMICS_CKPT := checkpoints/dynamics_v1.pt
LAPLACE_CKPT := checkpoints/laplace_v1.pt
RUNS := runs
EVAL := $(RUNS)/eval

all: manifest dynamics laplace train eval aggregate composite

manifest:
	$(PY) scripts/build_manifest.py $(IMAGENET_VAL) > $(MANIFEST)

smoke-manifest:
	$(PY) scripts/build_manifest.py $(IMAGENET_VAL) --limit 1000 > $(SMOKE_MANIFEST)

dynamics:
	$(PY) -m foveated.algos.dynamics_train --manifest $(MANIFEST) --out $(DYNAMICS_CKPT)

laplace:
	$(PY) -m foveated.experiments.fit_laplace --manifest $(MANIFEST) \
	  --dynamics-ckpt $(DYNAMICS_CKPT) --out $(LAPLACE_CKPT)

train:
	@for seed in $(SEEDS); do \
	  for agent in $(AGENTS); do \
	    $(PY) -m foveated.experiments.train_agent --agent $$agent --seed $$seed \
	      --manifest $(MANIFEST) --dynamics-ckpt $(DYNAMICS_CKPT) \
	      --laplace-ckpt $(LAPLACE_CKPT) --out-dir $(RUNS)/seed_$$seed ; \
	  done ; \
	done

eval:
	@for seed in $(SEEDS); do \
	  $(PY) -m foveated.experiments.evaluate --manifest $(MANIFEST) \
	    --ckpt-dir $(RUNS)/seed_$$seed --dynamics-ckpt $(DYNAMICS_CKPT) \
	    --out-dir $(EVAL)/seed_$$seed --seed $$seed ; \
	done

calibrate:
	$(PY) -m foveated.experiments.calibration --manifest $(MANIFEST) \
	  --dynamics-ckpt $(DYNAMICS_CKPT) --out-dir $(EVAL)

severity:
	$(PY) -m foveated.experiments.severity_sweep --manifest $(MANIFEST) \
	  --dynamics-ckpt $(DYNAMICS_CKPT) --out-dir $(EVAL)

feature-ablation:
	$(PY) -m foveated.experiments.feature_ablation --manifest $(MANIFEST) \
	  --agent-c-ckpt $(RUNS)/seed_42/agent_C_seed42_final.zip \
	  --dynamics-ckpt $(DYNAMICS_CKPT) --out $(EVAL)/feature_ablation.json

oracle:
	$(PY) -m foveated.experiments.oracle_topk --manifest $(MANIFEST) \
	  --out $(EVAL)/oracle_topk.csv

policy-viz:
	$(PY) -m foveated.experiments.policy_viz --manifest $(MANIFEST) \
	  --ckpt-dir $(RUNS)/seed_42 --dynamics-ckpt $(DYNAMICS_CKPT) \
	  --out $(EVAL)/policy_viz.pdf

aggregate:
	$(PY) -m foveated.experiments.aggregate_seeds --eval-root $(EVAL) --out-dir $(EVAL)

composite:
	$(PY) -m foveated.experiments.composite_figure --eval-dir $(EVAL) \
	  --out $(EVAL)/composite_figure.pdf

test:
	pytest -x -q tests/

lint:
	ruff check src/ tests/ scripts/

clean:
	rm -rf $(RUNS) $(EVAL) checkpoints/*.pt $(MANIFEST) $(SMOKE_MANIFEST)
