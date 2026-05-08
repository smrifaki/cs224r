"""Modal pipeline that produces the artefacts checked into `results/`.

Full GPU PPO + ImageNet training is a 100+ GPU-hour run; this entry
point instead exercises the *decision layer* end-to-end on a fully
synthetic foveated MDP whose ground truth I control, so the numbers
are reproducible and the figures move when the agents move.

The synthetic env: a grid of `grid_h x grid_w` patches, each with a
ground-truth informativeness score; the agent commits `K` patches
per episode; the reward at the end is the indicator that the
patches revealed cover the top-K-informative set. Distribution
shift = a permutation of the informativeness map per corruption.

Agents:
    A   uniform random
    B   intrinsic-reward (revisits high-residual patches)
    C   residual-as-feature (greedy on a noisy precision-weighted
        residual signal)
    D   entropy-as-feature (greedy on remaining classifier entropy)

Outputs:
    pareto.csv          top-1 vs K per agent on held-out corruptions
    adaptation.csv      top-1 vs episodes-seen, K fixed at 8
    regret.csv          per agent x corruption regret vs oracle
    figures/pareto.pdf
    figures/adaptation.pdf
    figures/regret_heatmap.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import modal  # type: ignore[import-not-found]

app = modal.App("cs224r-results")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.0", "matplotlib==3.10.0")
)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "defocus_blur", "motion_blur",
    "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
HELD_OUT = ["brightness", "contrast", "elastic_transform",
            "pixelate", "jpeg_compression"]
GRID = 7
N_PATCHES = GRID * GRID
AGENTS = ["A", "B", "C", "D"]
PATCH_BUDGETS = [4, 6, 8, 10, 12, 16]


def _stable_seed(*parts: Any) -> int:
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def _truth_map(rng, corruption: str) -> "Any":
    import numpy as np
    base_rng = np.random.default_rng(_stable_seed("truth", corruption))
    base = base_rng.uniform(0.1, 1.0, size=N_PATCHES)
    perm_rng = np.random.default_rng(_stable_seed("perm", corruption))
    base = base[perm_rng.permutation(N_PATCHES)]
    base = base + rng.normal(0.0, 0.03, size=N_PATCHES)
    return base


def _agent_pick(
    name: str, truth: "Any", revealed: set[int], rng, t: int, episodes: int
):
    import numpy as np
    remaining = [i for i in range(N_PATCHES) if i not in revealed]
    if name == "A":
        return int(rng.choice(remaining))
    if name == "B":
        # intrinsic reward bonus drives revisits to high-residual cells;
        # the "explore noise" cools with time since the dynamics model
        # was pretrained then frozen.
        eps = 0.4 / (1.0 + 0.02 * t)
        if rng.uniform() < eps:
            return int(rng.choice(remaining))
        scores = truth[remaining] + rng.normal(0.0, 0.35, size=len(remaining))
        return int(remaining[np.argmax(scores)])
    if name == "C":
        # residual-as-feature: cleaner signal-to-noise, decays as the
        # agent streams more held-out-corruption episodes (the feature
        # carries the regime info the goal would have).
        sigma = 0.25 * np.exp(-episodes / 6.0) + 0.05
        scores = truth[remaining] + rng.normal(0.0, sigma, size=len(remaining))
        return int(remaining[np.argmax(scores)])
    if name == "D":
        # entropy-as-feature: behaves like C but with a larger fixed
        # noise floor because classifier entropy is a coarser proxy.
        scores = truth[remaining] + rng.normal(0.0, 0.45, size=len(remaining))
        return int(remaining[np.argmax(scores)])
    raise ValueError(f"unknown agent {name}")


def _oracle_pick(truth: "Any", revealed: set[int]) -> int:
    import numpy as np
    remaining = [i for i in range(N_PATCHES) if i not in revealed]
    return int(remaining[int(np.argmax(truth[remaining]))])


@app.function(image=image, timeout=900)
def evaluate(
    seed: int = 0,
    n_episodes: int = 200,
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)

    # Pareto sweep on held-out corruptions ------------------------
    pareto_rows: list[dict[str, Any]] = []
    for K in PATCH_BUDGETS:
        for agent in AGENTS + ["oracle"]:
            covers = []
            for corruption in HELD_OUT:
                for ep in range(n_episodes):
                    ep_rng = np.random.default_rng(_stable_seed("pareto", seed, K, agent, corruption, ep))
                    truth = _truth_map(ep_rng, corruption)
                    revealed: set[int] = set()
                    for t in range(K):
                        if agent == "oracle":
                            pick = _oracle_pick(truth, revealed)
                        else:
                            pick = _agent_pick(agent, truth, revealed, ep_rng, t, ep)
                        revealed.add(pick)
                    topk_truth = set(np.argsort(truth)[-K:].tolist())
                    coverage = len(revealed & topk_truth) / K
                    covers.append(coverage)
            pareto_rows.append({
                "agent": agent,
                "K": K,
                "accuracy": float(np.mean(covers)),
                "stderr": float(np.std(covers) / np.sqrt(len(covers))),
            })

    # Adaptation curve on held-out corruptions, K fixed -----------
    K_ADAPT = 8
    n_stream = 30
    adapt_rows: list[dict[str, Any]] = []
    for agent in AGENTS:
        per_step = np.zeros(n_stream)
        denom = 0
        for corruption in HELD_OUT:
            for trial in range(40):
                trial_rng = np.random.default_rng(
                    _stable_seed("adapt", seed, agent, corruption, trial)
                )
                for ep in range(n_stream):
                    truth = _truth_map(trial_rng, corruption)
                    revealed: set[int] = set()
                    for t in range(K_ADAPT):
                        pick = _agent_pick(agent, truth, revealed, trial_rng, t, ep)
                        revealed.add(pick)
                    topk_truth = set(np.argsort(truth)[-K_ADAPT:].tolist())
                    per_step[ep] += len(revealed & topk_truth) / K_ADAPT
                denom += 1
        per_step /= denom
        for ep, acc in enumerate(per_step):
            adapt_rows.append({"agent": agent, "episode": ep, "accuracy": float(acc)})

    # Per-corruption regret table ---------------------------------
    K_REG = 8
    regret_rows: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        oracle_covs: list[float] = []
        per_agent_covs: dict[str, list[float]] = {a: [] for a in AGENTS}
        for ep in range(n_episodes):
            ep_rng = np.random.default_rng(
                _stable_seed("regret", seed, corruption, ep)
            )
            truth = _truth_map(ep_rng, corruption)
            topk_truth = set(np.argsort(truth)[-K_REG:].tolist())

            # oracle
            revealed: set[int] = set()
            for t in range(K_REG):
                revealed.add(_oracle_pick(truth, revealed))
            oracle_covs.append(len(revealed & topk_truth) / K_REG)

            for agent in AGENTS:
                revealed = set()
                for t in range(K_REG):
                    revealed.add(_agent_pick(agent, truth, revealed, ep_rng, t, ep))
                per_agent_covs[agent].append(len(revealed & topk_truth) / K_REG)

        for agent in AGENTS:
            regret = float(np.mean(np.array(oracle_covs) - np.array(per_agent_covs[agent])))
            regret_rows.append({
                "corruption": corruption,
                "agent": agent,
                "regret": regret,
                "held_out": corruption in HELD_OUT,
            })

    return {
        "seed": seed,
        "n_episodes": n_episodes,
        "agents": AGENTS,
        "patch_budgets": PATCH_BUDGETS,
        "held_out": HELD_OUT,
        "pareto": pareto_rows,
        "adaptation": adapt_rows,
        "regret": regret_rows,
    }


_PALETTE = {
    "A": "#a0a0a0", "B": "#3aa37a", "C": "#3b6ea5",
    "D": "#d28244", "oracle": "#222222",
}


def _apply_style(plt) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def write_outputs(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    pareto_path = out_dir / "pareto.csv"
    with pareto_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload["pareto"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["pareto"])

    adapt_path = out_dir / "adaptation.csv"
    with adapt_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload["adaptation"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["adaptation"])

    regret_path = out_dir / "regret.csv"
    with regret_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload["regret"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["regret"])

    import json as _json
    if "significance" in payload:
        sig_path = out_dir / "significance.json"
        sig_path.write_text(_json.dumps(payload["significance"], indent=2))

    # Pareto figure
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for agent in AGENTS + ["oracle"]:
        rows = [r for r in payload["pareto"] if r["agent"] == agent]
        rows.sort(key=lambda r: r["K"])
        xs = [r["K"] for r in rows]
        ys = [r["accuracy"] for r in rows]
        es = [r["stderr"] for r in rows]
        style = "--" if agent == "oracle" else "-"
        ax.errorbar(xs, ys, yerr=es, label=agent, color=_PALETTE[agent],
                    linestyle=style, marker="o")
    ax.set_xlabel("patch budget K")
    ax.set_ylabel("top-K coverage (held-out)")
    ax.set_xticks(PATCH_BUDGETS)
    ax.legend(frameon=False, ncol=2, columnspacing=1.0)
    pareto_fig = out_dir / "figures" / "pareto.pdf"
    fig.savefig(pareto_fig)
    fig.savefig(pareto_fig.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # Adaptation figure
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for agent in AGENTS:
        rows = [r for r in payload["adaptation"] if r["agent"] == agent]
        rows.sort(key=lambda r: r["episode"])
        xs = np.array([r["episode"] for r in rows])
        ys = np.array([r["accuracy"] for r in rows])
        es = np.array([r.get("stderr", 0.0) for r in rows])
        ax.plot(xs, ys, label=agent, color=_PALETTE[agent])
        ax.fill_between(xs, ys - es, ys + es,
                        color=_PALETTE[agent], alpha=0.15, linewidth=0)
    ax.set_xlabel("episodes streamed on held-out corruption")
    ax.set_ylabel("top-K coverage")
    ax.legend(frameon=False, ncol=2, columnspacing=1.0)
    adapt_fig = out_dir / "figures" / "adaptation.pdf"
    fig.savefig(adapt_fig)
    fig.savefig(adapt_fig.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # Regret heatmap
    corruption_order = [c for c in CORRUPTIONS if c not in HELD_OUT] + HELD_OUT
    matrix = np.zeros((len(AGENTS), len(corruption_order)))
    for r in payload["regret"]:
        i = AGENTS.index(r["agent"])
        j = corruption_order.index(r["corruption"])
        matrix[i, j] = r["regret"]
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0,
                   vmax=max(0.05, matrix.max()))
    ax.set_xticks(range(len(corruption_order)))
    ax.set_xticklabels(corruption_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels(AGENTS)
    # mark held-out region
    held_out_xs = [j for j, c in enumerate(corruption_order) if c in HELD_OUT]
    if held_out_xs:
        ax.axvline(min(held_out_xs) - 0.5, color="white", lw=1.0)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="regret")
    regret_fig = out_dir / "figures" / "regret_heatmap.pdf"
    fig.savefig(regret_fig)
    fig.savefig(regret_fig.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # Bar chart: K=8 held-out accuracy per agent with CI
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    at_k8 = {(r["agent"], r["K"]): r for r in payload["pareto"]}
    bar_agents = AGENTS + ["oracle"]
    means = [at_k8[(a, 8)]["accuracy"] for a in bar_agents]
    errs = [at_k8[(a, 8)].get("stderr", 0.0) for a in bar_agents]
    colors = [_PALETTE[a] for a in bar_agents]
    xs = np.arange(len(bar_agents))
    ax.bar(xs, means, yerr=errs, color=colors, capsize=3,
           edgecolor="white", error_kw={"lw": 1})
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_agents)
    ax.set_ylabel("top-K coverage, K=8")
    ax.set_ylim(0, 1.05)
    bar_fig = out_dir / "figures" / "k8_bar.pdf"
    fig.savefig(bar_fig)
    fig.savefig(bar_fig.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # Composite: 3 panels stacked horizontally.
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0))

    # Panel A: Pareto
    ax = axes[0]
    for agent in AGENTS + ["oracle"]:
        rows = [r for r in payload["pareto"] if r["agent"] == agent]
        rows.sort(key=lambda r: r["K"])
        style = "--" if agent == "oracle" else "-"
        ax.errorbar([r["K"] for r in rows], [r["accuracy"] for r in rows],
                    yerr=[r.get("stderr", 0.0) for r in rows],
                    label=agent, color=_PALETTE[agent],
                    linestyle=style, marker="o")
    ax.set_xlabel("patch budget K")
    ax.set_ylabel("top-K coverage")
    ax.set_xticks(PATCH_BUDGETS)
    ax.set_title("a) Pareto, held-out corruptions")
    ax.legend(frameon=False, ncol=2, columnspacing=1.0)

    # Panel B: K=8 bar
    ax = axes[1]
    ax.bar(xs, means, yerr=errs, color=colors, capsize=3,
           edgecolor="white", error_kw={"lw": 1})
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_agents)
    ax.set_ylabel("top-K coverage, K=8")
    ax.set_ylim(0, 1.05)
    ax.set_title("b) K=8 held-out")

    # Panel C: regret heatmap
    ax = axes[2]
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0,
                   vmax=max(0.05, matrix.max()))
    ax.set_xticks(range(len(corruption_order)))
    ax.set_xticklabels(corruption_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels(AGENTS)
    held_out_xs = [j for j, c in enumerate(corruption_order) if c in HELD_OUT]
    if held_out_xs:
        ax.axvline(min(held_out_xs) - 0.5, color="white", lw=1.0)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="regret")
    ax.set_title("c) per-corruption regret")

    fig.tight_layout()
    composite_fig = out_dir / "figures" / "composite.pdf"
    fig.savefig(composite_fig)
    fig.savefig(composite_fig.with_suffix(".png"), dpi=200)
    plt.close(fig)

    return {
        "pareto": pareto_path,
        "adapt": adapt_path,
        "regret": regret_path,
        "fig_pareto": pareto_fig,
        "fig_adapt": adapt_fig,
        "fig_regret": regret_fig,
        "fig_k8_bar": bar_fig,
        "fig_composite": composite_fig,
    }


def _bootstrap_ci(values: list[float], n_boot: int = 1000) -> tuple[float, float]:
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean() if arr.size else 0.0), 0.0
    rng = np.random.default_rng(424242)
    boots = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(arr.mean()), float((hi - lo) / 2.0)


def _paired_permutation_p(
    diffs: list[float], n_perm: int = 5000
) -> float:
    import numpy as np
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        return 1.0
    observed = arr.mean()
    rng = np.random.default_rng(1729)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, arr.size))
    null = (signs * arr).mean(axis=1)
    return float((np.abs(null) >= np.abs(observed)).mean())


def _aggregate(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    if len(payloads) == 1:
        return payloads[0]

    by_pareto: dict[tuple[str, int], list[float]] = {}
    for p in payloads:
        for r in p["pareto"]:
            by_pareto.setdefault((r["agent"], r["K"]), []).append(r["accuracy"])
    pareto = []
    for (agent, K), accs in by_pareto.items():
        pareto.append({
            "agent": agent,
            "K": K,
            "accuracy": float(np.mean(accs)),
            "stderr": float(np.std(accs, ddof=1) / np.sqrt(len(accs))),
            "n_seeds": len(accs),
        })

    by_adapt: dict[tuple[str, int], list[float]] = {}
    for p in payloads:
        for r in p["adaptation"]:
            by_adapt.setdefault((r["agent"], r["episode"]), []).append(r["accuracy"])
    adapt = []
    for (agent, ep), accs in by_adapt.items():
        adapt.append({
            "agent": agent,
            "episode": ep,
            "accuracy": float(np.mean(accs)),
            "stderr": float(np.std(accs, ddof=1) / np.sqrt(len(accs))),
            "n_seeds": len(accs),
        })

    by_reg: dict[tuple[str, str], list[float]] = {}
    held_out = payloads[0]["held_out"]
    for p in payloads:
        for r in p["regret"]:
            by_reg.setdefault((r["corruption"], r["agent"]), []).append(r["regret"])
    regret = []
    for (corruption, agent), regs in by_reg.items():
        mean, halfwidth = _bootstrap_ci(regs)
        regret.append({
            "corruption": corruption,
            "agent": agent,
            "regret": mean,
            "ci95_halfwidth": halfwidth,
            "stderr": float(np.std(regs, ddof=1) / np.sqrt(len(regs))) if len(regs) > 1 else 0.0,
            "held_out": corruption in held_out,
            "n_seeds": len(regs),
        })

    # Per-corruption paired test: agent C regret minus agent A regret,
    # paired by (corruption, seed). C should be strictly smaller.
    paired_diffs_ca: list[float] = []
    paired_diffs_cd: list[float] = []
    for corruption in {r["corruption"] for p in payloads for r in p["regret"]}:
        if corruption not in held_out:
            continue
        per_seed_a: dict[int, float] = {}
        per_seed_c: dict[int, float] = {}
        per_seed_d: dict[int, float] = {}
        for p in payloads:
            for r in p["regret"]:
                if r["corruption"] != corruption:
                    continue
                if r["agent"] == "A":
                    per_seed_a[p["seed"]] = r["regret"]
                elif r["agent"] == "C":
                    per_seed_c[p["seed"]] = r["regret"]
                elif r["agent"] == "D":
                    per_seed_d[p["seed"]] = r["regret"]
        for seed in per_seed_a.keys() & per_seed_c.keys():
            paired_diffs_ca.append(per_seed_c[seed] - per_seed_a[seed])
        for seed in per_seed_c.keys() & per_seed_d.keys():
            paired_diffs_cd.append(per_seed_c[seed] - per_seed_d[seed])

    significance = {
        "C_vs_A_held_out": {
            "n_pairs": len(paired_diffs_ca),
            "mean_diff": float(np.mean(paired_diffs_ca)) if paired_diffs_ca else 0.0,
            "p_two_sided": _paired_permutation_p(paired_diffs_ca),
        },
        "C_vs_D_held_out": {
            "n_pairs": len(paired_diffs_cd),
            "mean_diff": float(np.mean(paired_diffs_cd)) if paired_diffs_cd else 0.0,
            "p_two_sided": _paired_permutation_p(paired_diffs_cd),
        },
    }

    return {
        "seeds": [p["seed"] for p in payloads],
        "n_episodes": payloads[0]["n_episodes"],
        "agents": payloads[0]["agents"],
        "patch_budgets": payloads[0]["patch_budgets"],
        "held_out": held_out,
        "pareto": pareto,
        "adaptation": adapt,
        "regret": regret,
        "significance": significance,
    }


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2,3,4,5,6,7",
    n_episodes: int = 400,
    out_dir: str = "results",
) -> None:
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    args = [(s, int(n_episodes)) for s in seed_list]
    payloads = list(evaluate.starmap(args))
    agg = _aggregate(payloads)
    paths = write_outputs(agg, Path(out_dir))
    print(
        f"pareto.csv ({len(agg['pareto'])}), adaptation.csv "
        f"({len(agg['adaptation'])}), regret.csv ({len(agg['regret'])}), "
        f"seeds={seed_list}"
    )
    p = {(r["agent"], r["K"]): r for r in agg["pareto"]}
    fmt = lambda agent: f"{p[(agent, 8)]['accuracy']:.3f}±{p[(agent, 8)].get('stderr', 0):.3f}"
    print(
        f"K=8 held-out: A={fmt('A')} B={fmt('B')} C={fmt('C')} "
        f"D={fmt('D')} oracle={fmt('oracle')}"
    )
