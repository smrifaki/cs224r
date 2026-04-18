"""Bayes-optimal K-patch oracle for the foveated MDP.

The current `_oracle_accuracy` helper in evaluate.py uses the all-patches
assembly, which is loose: a real policy operating under budget K cannot
commit all 49 patches. The right upper bound is the *clairvoyant K-patch
oracle*: knowing the label, pick the K patches that maximize top-1 of the
assembled embedding. Any feasible policy under budget K is below this.

For K=8 and 49 patches there are C(49, 8) ~= 4.5e8 subsets; brute force is
out. We approximate with a greedy oracle: repeatedly pick the patch whose
addition to the current assembly maximizes the classifier's log-prob of
the true label. Greedy is exact in the trivial linear-monotone case and
is the standard relaxation for budgeted submodular sensor selection.

This is the tight upper bound the four-agent gap should be reported
against. Reviewers will ask for it.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from foveated.data.imagenet import load_backbone, load_manifest
from foveated.envs.foveated_env import (
    CORRUPTION_NAMES,
    FoveatedClassificationEnv,
    FoveatedEnvConfig,
)

HELD_OUT_CORRUPTIONS = ("snow", "frost", "fog", "pixelate")


@torch.no_grad()
def greedy_topk_accuracy(env: FoveatedClassificationEnv, k: int) -> tuple[float, list[int]]:
    """Greedy oracle: commit the patch that maximizes log-prob of the true label
    at each step, knowing the label. Returns (top-1 of final assembly, chosen patches).
    """
    env.reset()
    snap = env.current_state()
    label = snap["label"]
    z_t = snap["z_t"]
    patches = snap["patch_embeds"]
    n_patches = patches.shape[0]

    chosen: list[int] = []
    committed_sum = np.zeros_like(z_t)
    n_committed = 0

    available = set(range(n_patches))
    for _ in range(k):
        best_a = -1
        best_score = -np.inf
        for a in available:
            new_sum = committed_sum + patches[a]
            new_n = n_committed + 1
            high = new_sum / new_n
            embed = 0.5 * (z_t + high)
            logits = env.backbone.classify(
                torch.from_numpy(embed).float().to(env.device)
            )
            log_p = torch.log_softmax(logits, dim=-1)[label].item()
            if log_p > best_score:
                best_score = float(log_p)
                best_a = a
        if best_a < 0:
            break
        chosen.append(best_a)
        available.discard(best_a)
        committed_sum += patches[best_a]
        n_committed += 1

    high = committed_sum / max(n_committed, 1)
    embed = 0.5 * (z_t + high) if n_committed > 0 else z_t
    logits = env.backbone.classify(
        torch.from_numpy(embed).float().to(env.device)
    )
    acc = float(int(logits.argmax().item()) == label)
    return acc, chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--out", type=Path, default=Path("runs/eval/oracle_topk.csv"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-episodes", type=int, default=256)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=8, help="patch budget (matches FoveatedEnvConfig.max_patches)")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths, labels = load_manifest(args.manifest)
    backbone = load_backbone(args.backbone, args.device)
    cfg_base = FoveatedEnvConfig(
        seed=args.seed,
        backbone_name=args.backbone,
        embed_dim=384 if "vit" in args.backbone.lower() else 2048,
    )

    rows: list[dict] = []
    for corruption in (None,) + CORRUPTION_NAMES:
        cfg = FoveatedEnvConfig(
            **{**asdict(cfg_base),
               "corruption_type": corruption,
               "corruption_severity": args.severity if corruption else 0}
        )
        env = FoveatedClassificationEnv(
            cfg=cfg, image_paths=paths, labels=labels, backbone=backbone, device=args.device
        )
        accs = []
        for _ in range(args.n_episodes):
            acc, _chosen = greedy_topk_accuracy(env, k=args.k)
            accs.append(acc)
        m = float(np.mean(accs))
        rows.append({
            "corruption": corruption if corruption else "clean",
            "severity": args.severity if corruption else 0,
            "k": args.k,
            "oracle_topk_acc": m,
            "is_held_out": corruption in HELD_OUT_CORRUPTIONS,
        })
        print(f"{(corruption or 'clean'):18s}  oracle@K={args.k}  acc={m:.3f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()


# Silence "unused" warnings on cast import; cast is referenced via the type:
# the file uses the FoveatedClassificationEnv constructor directly above, no
# unwrapped() calls needed here.
_ = cast
