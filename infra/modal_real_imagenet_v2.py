"""Real cs224r training v2 — decoupled obs path / classifier path.

Bug fix vs v1: classifier head now receives the **full 384-dim
mask-pooled feature**, not a zero-padded 8-dim slice. The PPO obs
stays compact (mask + residual + entropy summary).

  - Backbone: pretrained timm vit_small_patch16_224 (frozen).
  - Head: linear(384, n_classes), fine-tuned 2 epochs on whole-image
    features.
  - Foveated MDP: 7x7 patches, K=8 reveals, classification reward
    = (top-1 acc) - patch_cost * K.
  - Agents A and D first (no dynamics model required). B and C
    follow once we add a forward-dynamics pretrain step.
  - 3 seeds, 60k env-steps per (agent, seed). 6 jobs in parallel.
"""
from __future__ import annotations

import modal


app = modal.App("cs224r-real-v2")

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
        "imagenet-c==0.0.3",
    )
)

data_vol = modal.Volume.from_name("cs224r-imagenet")
ckpt_vol = modal.Volume.from_name("cs224r-ckpts", create_if_missing=True)


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
    backbone_epochs: int    = 2,
) -> dict:
    import time
    from pathlib import Path

    import gymnasium as gym
    import numpy as np
    import timm
    import torch
    import torch.nn.functional as F
    import wandb
    from datasets import load_from_disk
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from torchvision import transforms

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data --------------------------------------------------------
    ds_train = load_from_disk("/data/imagenet/train")
    ds_val   = load_from_disk("/data/imagenet/validation")
    rng_classes = np.random.default_rng(0)
    selected = sorted(rng_classes.choice(1000, size=n_classes, replace=False).tolist())
    cls_to_local = {c: i for i, c in enumerate(selected)}

    print(f"sampling {images_per_class_train}/class train, "
          f"{images_per_class_val}/class val...")
    train_pairs = []
    val_pairs   = []
    labels_t = np.array(ds_train["label"])
    labels_v = np.array(ds_val["label"])
    for c in selected:
        idx_t = np.where(labels_t == c)[0]
        np_rng.shuffle(idx_t)
        for i in idx_t[:images_per_class_train]:
            train_pairs.append((int(i), cls_to_local[c]))
        idx_v = np.where(labels_v == c)[0]
        np_rng.shuffle(idx_v)
        for i in idx_v[:images_per_class_val]:
            val_pairs.append((int(i), cls_to_local[c]))
    print(f"  train={len(train_pairs)}, val={len(val_pairs)}")

    # ---- backbone + head --------------------------------------------
    print("loading vit_small_patch16_224 (pretrained, frozen)...")
    bb = timm.create_model("vit_small_patch16_224", pretrained=True,
                           num_classes=0).to(device).eval()
    for p in bb.parameters():
        p.requires_grad = False
    feat_dim = bb.num_features  # 384

    head = torch.nn.Linear(feat_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    norm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def to_tensor(idx, ds):
        return norm(ds[idx]["image"].convert("RGB"))

    print(f"pretraining classifier head ({backbone_epochs} epochs)...")
    t0 = time.time()
    BATCH = 64
    for epoch in range(backbone_epochs):
        head.train()
        rng_perm = np_rng.permutation(len(train_pairs))
        for i in range(0, len(train_pairs), BATCH):
            batch_idx = rng_perm[i:i + BATCH]
            xs = torch.stack([to_tensor(train_pairs[j][0], ds_train)
                             for j in batch_idx]).to(device)
            ys = torch.tensor([train_pairs[j][1] for j in batch_idx],
                              device=device)
            with torch.no_grad():
                feats = bb(xs)
            logits = head(feats)
            loss = F.cross_entropy(logits, ys)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
    print(f"  trained in {time.time()-t0:.1f}s")

    # backbone val acc
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(val_pairs), BATCH):
            xs = torch.stack([to_tensor(val_pairs[j][0], ds_val)
                             for j in range(i, min(i + BATCH, len(val_pairs)))]).to(device)
            ys = torch.tensor([val_pairs[j][1]
                               for j in range(i, min(i + BATCH, len(val_pairs)))],
                              device=device)
            correct += (head(bb(xs)).argmax(-1) == ys).sum().item()
            total += ys.numel()
    bb_acc = correct / total
    print(f"  backbone head val acc: {bb_acc:.3f}")

    # ---- per-patch features cache -----------------------------------
    GRID = 7
    N_PATCHES = GRID * GRID
    K = 8
    PATCH_COST = 0.01

    print("caching per-patch ViT features for the full image set...")

    def patch_feats(ds, pairs):
        feats_per = []
        labels_per = []
        bb.eval()
        with torch.no_grad():
            for i in range(0, len(pairs), BATCH):
                batch = pairs[i:i + BATCH]
                xs = torch.stack([to_tensor(p[0], ds) for p in batch]).to(device)
                # vit_small forward_features returns (B, 197, 384) for
                # patch_size 16: 1 CLS + 196 patches (14x14).
                tokens = bb.forward_features(xs)
                patch_tokens = tokens[:, 1:, :]  # drop CLS
                B = patch_tokens.shape[0]
                # 14x14 patch grid -> pool to 7x7 by 2x2 mean
                g14 = patch_tokens.reshape(B, 14, 14, feat_dim)
                g7  = g14.reshape(B, 7, 2, 7, 2, feat_dim).mean(dim=(2, 4))
                g49 = g7.reshape(B, N_PATCHES, feat_dim).cpu().numpy()
                for j, _ in enumerate(batch):
                    feats_per.append(g49[j])
                    labels_per.append(batch[j][1])
        return feats_per, labels_per

    train_feats, train_labels = patch_feats(ds_train, train_pairs)
    val_feats,   val_labels   = patch_feats(ds_val,   val_pairs)
    print(f"  cached: {len(train_feats)} train, {len(val_feats)} val")

    # ---- foveated env: decoupled obs / classifier -------------------
    class FoveatedEnv(gym.Env):
        """Obs: mask (49) + per-patch residual (49) + assembly entropy (1)
        Reward: 1 if final classification correct, minus K * patch_cost.
        Classifier head: full 384-dim mask-pooled features.
        """
        def __init__(self, feats, labels):
            super().__init__()
            self.feats  = feats
            self.labels = labels
            obs_dim = N_PATCHES + N_PATCHES + 1
            self.action_space = spaces.Discrete(N_PATCHES)
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            i = int(np_rng.integers(len(self.feats)))
            self.full = self.feats[i].astype(np.float32)      # (49, 384)
            self.label = int(self.labels[i])
            self.mask = np.zeros(N_PATCHES, dtype=np.float32)
            self.residual = np.zeros(N_PATCHES, dtype=np.float32)
            self.k = 0
            self.feat_buf = np.zeros((N_PATCHES, feat_dim), dtype=np.float32)
            return self._obs(), {}

        def _entropy_of_assembly(self):
            mask = self.mask[:, None]
            denom = max(1.0, float(self.mask.sum()))
            pooled = (self.feat_buf * mask).sum(0) / denom  # (384,)
            with torch.no_grad():
                logits = head(torch.from_numpy(pooled).to(device))
                probs = torch.softmax(logits, dim=-1)
                ent = float((-probs * torch.log(probs + 1e-9)).sum().item())
            return ent

        def _obs(self):
            ent = self._entropy_of_assembly()
            return np.concatenate([
                self.mask,
                self.residual,
                np.array([ent], dtype=np.float32),
            ]).astype(np.float32)

        def _predict(self):
            mask = self.mask[:, None]
            denom = max(1.0, float(self.mask.sum()))
            pooled = (self.feat_buf * mask).sum(0) / denom
            with torch.no_grad():
                logits = head(torch.from_numpy(pooled).to(device))
            return int(logits.argmax().item())

        def step(self, action):
            action = int(action)
            terminated = truncated = False
            reward = 0.0
            if self.mask[action] == 1.0:
                terminated = True
            else:
                feat = self.full[action]
                pre_mean = self.feat_buf.sum(0) / max(1.0, float(self.mask.sum()))
                self.residual[action] = float(np.linalg.norm(feat - pre_mean) ** 2)
                self.feat_buf[action] = feat
                self.mask[action] = 1.0
                self.k += 1
                reward -= PATCH_COST
            if self.k >= K:
                truncated = True
                pred = self._predict()
                reward += 1.0 if pred == self.label else 0.0
            info = {"label": self.label,
                    "pred":  self._predict() if (terminated or truncated) else -1}
            return self._obs(), reward, terminated, truncated, info

    # Agent variants override the residual channel
    if agent == "A":
        class A_(FoveatedEnv):
            def _obs(self):
                ent = self._entropy_of_assembly()
                return np.concatenate([
                    self.mask,
                    np.zeros(N_PATCHES, dtype=np.float32),
                    np.array([ent], dtype=np.float32),
                ]).astype(np.float32)
        env_cls = A_
    elif agent == "D":
        # D already uses entropy in obs; residual channel zeroed so
        # the difference vs A is entropy presence as a richer signal.
        # We instead let D's "residual" be the entropy spread across
        # patches (placeholder; entropy is global).
        class D_(FoveatedEnv):
            def _obs(self):
                ent = self._entropy_of_assembly()
                ent_per_patch = np.full(N_PATCHES, ent, dtype=np.float32)
                return np.concatenate([
                    self.mask,
                    ent_per_patch * 0.1,
                    np.array([ent], dtype=np.float32),
                ]).astype(np.float32)
        env_cls = D_
    else:
        raise ValueError(f"agent {agent} requires dynamics; not in this run")

    env = DummyVecEnv([lambda: env_cls(train_feats, train_labels)])

    run = wandb.init(
        project="cs224r-foveated-v2",
        job_type="train",
        tags=["real", "imagenet100", f"agent_{agent}", f"seed_{seed}"],
        config={
            "agent": agent, "seed": seed, "n_classes": n_classes,
            "n_env_steps": n_env_steps,
            "backbone_val_acc": bb_acc,
            "K": K, "GRID": GRID,
            "images_per_class_train": images_per_class_train,
            "feat_dim": feat_dim,
        },
    )

    class LogCb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rs, self.accs = [], []

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if info.get("pred", -1) >= 0:
                    correct = 1.0 if info["pred"] == info["label"] else 0.0
                    self.accs.append(correct)
            if len(self.accs) and len(self.accs) % 100 == 0:
                wandb.log({
                    "train/mean_acc_last_500":
                        float(np.mean(self.accs[-500:])),
                    "train/n_terminal_episodes": len(self.accs),
                })
            return True

    print(f"starting PPO: agent={agent} seed={seed} n_env_steps={n_env_steps}")
    model = PPO(
        "MlpPolicy", env, verbose=0, seed=seed,
        n_steps=512, batch_size=64, learning_rate=3e-4,
        policy_kwargs={"net_arch": [256, 256]},
    )
    cb = LogCb()
    t0 = time.time()
    model.learn(total_timesteps=n_env_steps, callback=cb, progress_bar=False)
    train_time = time.time() - t0
    print(f"trained in {train_time:.1f}s, terminal-eps={len(cb.accs)}")

    # ---- eval -------------------------------------------------------
    eval_env = env_cls(val_feats, val_labels)
    eval_rewards, eval_accs = [], []
    for _ in range(400):
        obs, _ = eval_env.reset()
        done = truncated = False
        total_r = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = eval_env.step(int(action))
            total_r += r
        eval_rewards.append(total_r)
        eval_accs.append(1.0 if info.get("pred") == info.get("label") else 0.0)
    eval_mean_reward = float(np.mean(eval_rewards))
    eval_mean_acc    = float(np.mean(eval_accs))

    wandb.log({
        "eval/mean_reward": eval_mean_reward,
        "eval/mean_acc":    eval_mean_acc,
        "eval/n":           len(eval_rewards),
        "train/wall_s":     train_time,
    })
    wandb.finish()

    print(f"agent={agent} seed={seed}  eval_acc={eval_mean_acc:.3f}  "
          f"reward={eval_mean_reward:.3f}")

    # Save policy
    out_dir = Path(f"/ckpt/v2/agent{agent}/seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "ppo")
    ckpt_vol.commit()

    return {
        "agent": agent, "seed": seed, "n_env_steps": n_env_steps,
        "n_classes": n_classes,
        "backbone_val_acc":   bb_acc,
        "train_time_s":       train_time,
        "n_terminal_eps":     len(cb.accs),
        "eval_mean_reward":   eval_mean_reward,
        "eval_mean_acc":      eval_mean_acc,
        "train_mean_acc_last500": (
            float(np.mean(cb.accs[-500:])) if len(cb.accs) >= 500 else float("nan")
        ),
        "wandb_url": run.url,
    }


@app.local_entrypoint()
def main(
    agents: str       = "A,D",
    seeds: str        = "0,1,2",
    n_env_steps: int  = 60_000,
    n_classes: int    = 100,
):
    import json
    from pathlib import Path

    agent_list = [a.strip() for a in agents.split(",") if a.strip()]
    seed_list  = [int(s) for s in seeds.split(",") if s.strip()]
    args = [(a, s, int(n_env_steps), int(n_classes), 60, 12, 2)
            for a in agent_list for s in seed_list]
    print(f"launching {len(args)} parallel T4 jobs: {agent_list} x {seed_list}")
    out = list(train_one.starmap(args))

    Path("/tmp/real_cs224r_v2_result.json").write_text(json.dumps(out, indent=2))
    for o in out:
        print(
            f"  agent={o['agent']} seed={o['seed']}  "
            f"bb={o['backbone_val_acc']:.3f}  "
            f"eval={o['eval_mean_acc']:.3f}  "
            f"train_last500={o['train_mean_acc_last500']:.3f}  "
            f"t={o['train_time_s']:.0f}s"
        )
