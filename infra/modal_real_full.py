"""Full proposal sweep on Modal T4.

  step 1: collect random rollouts on the foveated env, save
          (z_t, action, z_{t+1}) triples.
  step 2: train a small forward dynamics model on those triples.
  step 3: train Agents A, B, C, D x 3 seeds (12 jobs in parallel)
          using the same backbone and dynamics checkpoint.

Agent definitions:
  A : no extra features.
  B : intrinsic reward = beta * ||residual||^2, dynamics frozen.
  C : per-patch precision-weighted residual fed into the obs.
  D : assembly entropy fed into the obs (no dynamics needed).
"""
from __future__ import annotations

import modal

app = modal.App("cs224r-real-full")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "timm==1.0.9",
        "gymnasium==0.29.1",
        "stable-baselines3==2.3.2",
        "numpy==1.26.4",
        "datasets==2.21.0",
        "wandb==0.18.3",
    )
)

data_vol = modal.Volume.from_name("cs224r-imagenet")
ckpt_vol = modal.Volume.from_name("cs224r-ckpts", create_if_missing=True)


# Shared utilities ------------------------------------------------------------


def _build_backbone_and_head(
    device: str, n_classes: int, train_pairs, ds_train, backbone_epochs: int,
    seed: int,
):
    import time
    import numpy as np
    import timm
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    np_rng = np.random.default_rng(seed)
    bb = timm.create_model("vit_small_patch16_224", pretrained=True,
                           num_classes=0).to(device).eval()
    for p in bb.parameters():
        p.requires_grad = False
    feat_dim = bb.num_features

    norm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def to_tensor(idx):
        return norm(ds_train[idx]["image"].convert("RGB"))

    head = torch.nn.Linear(feat_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    BATCH = 64
    print(f"  pretraining classifier head x{backbone_epochs} epochs...")
    t0 = time.time()
    for _ in range(backbone_epochs):
        head.train()
        perm = np_rng.permutation(len(train_pairs))
        for i in range(0, len(train_pairs), BATCH):
            idx = perm[i:i + BATCH]
            xs = torch.stack([to_tensor(train_pairs[j][0]) for j in idx]).to(device)
            ys = torch.tensor([train_pairs[j][1] for j in idx], device=device)
            with torch.no_grad():
                feats = bb(xs)
            logits = head(feats)
            loss = F.cross_entropy(logits, ys)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
    print(f"    done in {time.time()-t0:.1f}s")

    return bb, head, norm, feat_dim


def _cache_patch_features(bb, ds, pairs, device, feat_dim, grid=7):
    import numpy as np
    import torch
    from torchvision import transforms

    norm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def to_tensor(idx):
        return norm(ds[idx]["image"].convert("RGB"))

    BATCH = 64
    N_PATCHES = grid * grid
    feats_per = []
    labels_per = []
    bb.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i + BATCH]
            xs = torch.stack([to_tensor(p[0]) for p in batch]).to(device)
            tokens = bb.forward_features(xs)
            patch_tokens = tokens[:, 1:, :]
            B = patch_tokens.shape[0]
            g14 = patch_tokens.reshape(B, 14, 14, feat_dim)
            g7  = g14.reshape(B, 7, 2, 7, 2, feat_dim).mean(dim=(2, 4))
            g49 = g7.reshape(B, N_PATCHES, feat_dim).cpu().numpy()
            for j, _ in enumerate(batch):
                feats_per.append(g49[j])
                labels_per.append(batch[j][1])
    return feats_per, labels_per


# Step 1: collect random rollouts + train dynamics ----------------------------


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def pretrain_dynamics(
    n_classes: int = 100,
    images_per_class_train: int = 60,
    n_random_episodes: int = 4000,
    dyn_epochs: int = 30,
) -> dict:
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_from_disk

    np_rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_train = load_from_disk("/data/imagenet/train")
    labels_t = np.array(ds_train["label"])
    rng_classes = np.random.default_rng(0)
    selected = sorted(rng_classes.choice(1000, size=n_classes, replace=False).tolist())
    cls_to_local = {c: i for i, c in enumerate(selected)}
    train_pairs = []
    for c in selected:
        idx_t = np.where(labels_t == c)[0]
        np_rng.shuffle(idx_t)
        for i in idx_t[:images_per_class_train]:
            train_pairs.append((int(i), cls_to_local[c]))
    print(f"  collected {len(train_pairs)} train images")

    bb, head, _, feat_dim = _build_backbone_and_head(
        device, n_classes, train_pairs, ds_train,
        backbone_epochs=2, seed=0,
    )

    train_feats, train_labels = _cache_patch_features(
        bb, ds_train, train_pairs, device, feat_dim,
    )
    GRID = 7
    N_PATCHES = GRID * GRID
    K = 8

    # ---- random-policy rollouts -------------------------------------
    print(f"collecting {n_random_episodes} random episodes ({K} steps each)...")
    triples_z = []
    triples_a = []
    triples_zn = []
    for ep in range(n_random_episodes):
        i = int(np_rng.integers(len(train_feats)))
        f = train_feats[i].astype(np.float32)
        mask = np.zeros(N_PATCHES, dtype=np.float32)
        feat_buf = np.zeros((N_PATCHES, feat_dim), dtype=np.float32)
        # pre-step state z = mask-pooled feat buf (zero at start)
        prev_z = np.zeros(feat_dim, dtype=np.float32)
        for _ in range(K):
            remaining = np.where(mask == 0)[0]
            if len(remaining) == 0:
                break
            action = int(np_rng.choice(remaining))
            feat_buf[action] = f[action]
            mask[action] = 1.0
            new_z = (feat_buf * mask[:, None]).sum(0) / mask.sum()
            triples_z.append(prev_z.copy())
            triples_a.append(action)
            triples_zn.append(new_z.copy())
            prev_z = new_z
    print(f"  collected {len(triples_z)} (z, a, z_next) triples")

    Z  = torch.from_numpy(np.stack(triples_z)).to(device)         # (N, 384)
    A  = torch.tensor(triples_a, device=device, dtype=torch.long)  # (N,)
    Zn = torch.from_numpy(np.stack(triples_zn)).to(device)        # (N, 384)
    print(f"  Z shape={tuple(Z.shape)}, A shape={tuple(A.shape)}")

    # ---- dynamics model ---------------------------------------------
    class ForwardDynamics(torch.nn.Module):
        def __init__(self, embed_dim=384, n_actions=49, hidden=256):
            super().__init__()
            self.embed_a = torch.nn.Embedding(n_actions, hidden)
            self.body = torch.nn.Sequential(
                torch.nn.Linear(embed_dim + hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
            )
            self.head_mu = torch.nn.Linear(hidden, embed_dim)
            self.head_log_s = torch.nn.Linear(hidden, embed_dim)

        def forward(self, z, a):
            h = self.body(torch.cat([z, self.embed_a(a)], dim=-1))
            return self.head_mu(h), self.head_log_s(h).clamp(-6.0, 4.0)

    model = ForwardDynamics(feat_dim, N_PATCHES, 256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    BATCH = 256

    print(f"training dynamics for {dyn_epochs} epochs over {len(Z)} samples...")
    t0 = time.time()
    history = []
    for epoch in range(dyn_epochs):
        perm = torch.randperm(len(Z), device=device)
        running = 0.0
        n_batches = 0
        for i in range(0, len(Z), BATCH):
            idx = perm[i:i + BATCH]
            mu, log_s = model(Z[idx], A[idx])
            # Gaussian NLL
            inv_s2 = torch.exp(-log_s)
            loss = (0.5 * ((mu - Zn[idx]) ** 2) * inv_s2 + 0.5 * log_s).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n_batches += 1
        history.append(running / n_batches)
        if epoch % 5 == 0 or epoch == dyn_epochs - 1:
            print(f"  epoch {epoch:>3d}  nll = {history[-1]:.4f}")
    print(f"  trained in {time.time()-t0:.1f}s")

    out_path = Path("/ckpt/dynamics_v1.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"embed_dim": feat_dim, "n_actions": N_PATCHES, "hidden": 256},
        "history": history,
    }, out_path)
    ckpt_vol.commit()
    print(f"  saved {out_path}")

    return {
        "n_triples": len(Z),
        "n_train_images": len(train_pairs),
        "feat_dim": feat_dim,
        "dynamics_history": history,
        "final_nll": history[-1],
    }


# Step 2: train one agent with the dynamics checkpoint ----------------------


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 90,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_one(
    agent: str,
    seed: int,
    n_env_steps: int = 60_000,
    n_classes: int = 100,
    images_per_class_train: int = 60,
    images_per_class_val:   int = 12,
) -> dict:
    import time
    from pathlib import Path

    import gymnasium as gym
    import numpy as np
    import torch
    import wandb
    from datasets import load_from_disk
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_train = load_from_disk("/data/imagenet/train")
    ds_val   = load_from_disk("/data/imagenet/validation")
    rng_classes = np.random.default_rng(0)
    selected = sorted(rng_classes.choice(1000, size=n_classes, replace=False).tolist())
    cls_to_local = {c: i for i, c in enumerate(selected)}
    train_pairs, val_pairs = [], []
    labels_t = np.array(ds_train["label"])
    labels_v = np.array(ds_val["label"])
    for c in selected:
        idx_t = np.where(labels_t == c)[0]; np_rng.shuffle(idx_t)
        for i in idx_t[:images_per_class_train]:
            train_pairs.append((int(i), cls_to_local[c]))
        idx_v = np.where(labels_v == c)[0]; np_rng.shuffle(idx_v)
        for i in idx_v[:images_per_class_val]:
            val_pairs.append((int(i), cls_to_local[c]))

    bb, head, _, feat_dim = _build_backbone_and_head(
        device, n_classes, train_pairs, ds_train,
        backbone_epochs=2, seed=seed,
    )

    correct = total = 0
    BATCH = 64
    from torchvision import transforms
    norm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        for i in range(0, len(val_pairs), BATCH):
            sub = val_pairs[i:i + BATCH]
            xs = torch.stack([norm(ds_val[p[0]]["image"].convert("RGB"))
                              for p in sub]).to(device)
            ys = torch.tensor([p[1] for p in sub], device=device)
            correct += (head(bb(xs)).argmax(-1) == ys).sum().item()
            total += ys.numel()
    bb_acc = correct / total
    print(f"backbone head val acc: {bb_acc:.3f}")

    train_feats, train_labels = _cache_patch_features(
        bb, ds_train, train_pairs, device, feat_dim,
    )
    val_feats, val_labels = _cache_patch_features(
        bb, ds_val, val_pairs, device, feat_dim,
    )

    GRID = 7
    N_PATCHES = GRID * GRID
    K = 8
    PATCH_COST = 0.01

    # Load dynamics ckpt for Agents B and C
    dyn = None
    if agent in ("B", "C"):
        ckpt = torch.load("/ckpt/dynamics_v1.pt", map_location=device)
        cfg = ckpt["config"]
        class ForwardDynamics(torch.nn.Module):
            def __init__(self, embed_dim, n_actions, hidden):
                super().__init__()
                self.embed_a = torch.nn.Embedding(n_actions, hidden)
                self.body = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim + hidden, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.ReLU(),
                )
                self.head_mu = torch.nn.Linear(hidden, embed_dim)
                self.head_log_s = torch.nn.Linear(hidden, embed_dim)

            def forward(self, z, a):
                h = self.body(torch.cat([z, self.embed_a(a)], dim=-1))
                return self.head_mu(h), self.head_log_s(h).clamp(-6.0, 4.0)
        dyn = ForwardDynamics(**cfg).to(device).eval()
        dyn.load_state_dict(ckpt["state_dict"])
        for p in dyn.parameters():
            p.requires_grad = False
        print("loaded dynamics checkpoint")

    INTRINSIC_BETA = 0.05

    class FoveatedEnv(gym.Env):
        def __init__(self, feats, labels):
            super().__init__()
            self.feats = feats; self.labels = labels
            obs_dim = N_PATCHES + N_PATCHES + 1
            self.action_space = spaces.Discrete(N_PATCHES)
            self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            i = int(np_rng.integers(len(self.feats)))
            self.full = self.feats[i].astype(np.float32)
            self.label = int(self.labels[i])
            self.mask = np.zeros(N_PATCHES, dtype=np.float32)
            self.residual = np.zeros(N_PATCHES, dtype=np.float32)
            self.k = 0
            self.feat_buf = np.zeros((N_PATCHES, feat_dim), dtype=np.float32)
            return self._obs(), {}

        def _pooled(self):
            denom = max(1.0, float(self.mask.sum()))
            return (self.feat_buf * self.mask[:, None]).sum(0) / denom

        def _entropy(self):
            with torch.no_grad():
                pooled = self._pooled()
                logits = head(torch.from_numpy(pooled).to(device))
                probs  = torch.softmax(logits, dim=-1)
                return float((-probs * torch.log(probs + 1e-9)).sum().item())

        def _residual_for(self, action):
            """Compute the precision-weighted residual prediction
            error for committing `action`. Uses the dynamics model
            if available, else falls back to ||f - running_mean||^2.
            """
            feat = self.full[action]
            if dyn is None:
                pre_mean = self.feat_buf.sum(0) / max(1.0, float(self.mask.sum()))
                return float(np.linalg.norm(feat - pre_mean) ** 2)
            with torch.no_grad():
                z = torch.from_numpy(self._pooled()).unsqueeze(0).to(device)
                a = torch.tensor([action], device=device, dtype=torch.long)
                mu, log_s = dyn(z, a)
                inv_s2 = torch.exp(-log_s)
                err = (torch.from_numpy(feat).to(device) - mu.squeeze(0))
                return float((err ** 2 * inv_s2.squeeze(0)).sum().item())

        def _obs(self):
            mask  = self.mask.astype(np.float32)
            resid = self.residual.astype(np.float32)
            ent   = self._entropy()
            if agent == "A":
                resid = np.zeros_like(resid)
            elif agent == "D":
                resid = np.full(N_PATCHES, ent * 0.1, dtype=np.float32)
            return np.concatenate([mask, resid, np.array([ent], dtype=np.float32)])

        def _predict(self):
            with torch.no_grad():
                pooled = self._pooled()
                logits = head(torch.from_numpy(pooled).to(device))
            return int(logits.argmax().item())

        def step(self, action):
            action = int(action)
            terminated = truncated = False
            reward = 0.0
            if self.mask[action] == 1.0:
                terminated = True
            else:
                # Compute residual BEFORE committing the patch
                residual = self._residual_for(action)
                self.residual[action] = residual
                self.feat_buf[action] = self.full[action]
                self.mask[action] = 1.0
                self.k += 1
                reward -= PATCH_COST
                if agent == "B":
                    reward += INTRINSIC_BETA * residual
            if self.k >= K:
                truncated = True
                pred = self._predict()
                reward += 1.0 if pred == self.label else 0.0
            info = {"label": self.label,
                    "pred":  self._predict() if (terminated or truncated) else -1}
            return self._obs(), reward, terminated, truncated, info

    env = DummyVecEnv([lambda: FoveatedEnv(train_feats, train_labels)])

    run = wandb.init(
        project="cs224r-foveated-full",
        job_type="train",
        tags=["real", "imagenet100", f"agent_{agent}", f"seed_{seed}"],
        config={
            "agent": agent, "seed": seed, "n_classes": n_classes,
            "n_env_steps": n_env_steps,
            "backbone_val_acc": bb_acc,
            "K": K, "GRID": GRID, "feat_dim": feat_dim,
            "intrinsic_beta": INTRINSIC_BETA if agent == "B" else 0.0,
        },
    )

    class LogCb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.accs = []

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if info.get("pred", -1) >= 0:
                    self.accs.append(1.0 if info["pred"] == info["label"] else 0.0)
            if len(self.accs) and len(self.accs) % 100 == 0:
                wandb.log({"train/mean_acc_last_500":
                            float(np.mean(self.accs[-500:])),
                           "train/n_terminal": len(self.accs)})
            return True

    print(f"PPO: agent={agent} seed={seed} n={n_env_steps}")
    model = PPO("MlpPolicy", env, verbose=0, seed=seed,
                n_steps=512, batch_size=64, learning_rate=3e-4,
                policy_kwargs={"net_arch": [256, 256]})
    cb = LogCb()
    t0 = time.time()
    model.learn(total_timesteps=n_env_steps, callback=cb, progress_bar=False)
    train_time = time.time() - t0

    eval_env = FoveatedEnv(val_feats, val_labels)
    eval_accs = []
    for _ in range(400):
        obs, _ = eval_env.reset()
        done = truncated = False
        while not (done or truncated):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = eval_env.step(int(a))
        eval_accs.append(1.0 if info.get("pred") == info.get("label") else 0.0)
    eval_mean_acc = float(np.mean(eval_accs))

    wandb.log({"eval/mean_acc": eval_mean_acc,
               "eval/n":         len(eval_accs),
               "train/wall_s":   train_time})
    wandb.finish()
    print(f"agent={agent} seed={seed}  eval_acc={eval_mean_acc:.3f}  "
          f"wall={train_time:.1f}s")

    out_dir = Path(f"/ckpt/full/agent{agent}/seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "ppo")
    ckpt_vol.commit()

    return {
        "agent": agent, "seed": seed, "n_env_steps": n_env_steps,
        "backbone_val_acc": bb_acc,
        "eval_mean_acc": eval_mean_acc,
        "train_time_s": train_time,
        "train_mean_acc_last500": (
            float(np.mean(cb.accs[-500:])) if len(cb.accs) >= 500 else float("nan")
        ),
        "wandb_url": run.url,
    }


@app.local_entrypoint()
def main(
    skip_dynamics: bool = False,
    agents: str = "A,B,C,D",
    seeds: str  = "0,1,2",
    n_env_steps: int = 60_000,
    n_classes: int   = 100,
):
    import json
    from pathlib import Path

    if not skip_dynamics:
        print("==> step 1: pretrain dynamics")
        dyn_out = pretrain_dynamics.remote(n_classes=n_classes)
        print(f"    final NLL: {dyn_out['final_nll']:.4f}")
    else:
        print("==> skipping dynamics pretrain (already on /ckpt)")

    agent_list = [a.strip() for a in agents.split(",") if a.strip()]
    seed_list  = [int(s) for s in seeds.split(",") if s.strip()]
    args = [(a, s, int(n_env_steps), int(n_classes), 60, 12)
            for a in agent_list for s in seed_list]
    print(f"==> step 2: training {len(args)} PPO jobs in parallel")
    out = list(train_one.starmap(args))

    Path("/tmp/real_cs224r_full_result.json").write_text(json.dumps(out, indent=2))
    print("\n--- summary ---")
    for o in out:
        print(f"  agent={o['agent']} seed={o['seed']}  "
              f"bb={o['backbone_val_acc']:.3f}  "
              f"eval={o['eval_mean_acc']:.3f}  "
              f"train500={o['train_mean_acc_last500']:.3f}  "
              f"t={o['train_time_s']:.0f}s")
