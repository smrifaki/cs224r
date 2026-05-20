"""Real cs224r training on Modal T4: ImageNet + ImageNet-C foveated PPO.

Uses the cs224r-imagenet Modal Volume (already has a HF-format
ImageNet sitting at /data/imagenet/{train,validation}).

  - Backbone: pretrained ViT-small from timm. Frozen feature
    extractor, classification head fine-tuned for 1 epoch on a
    100-class ImageNet subset.
  - Foveated env: 7x7 patch grid over 224x224, K=8 reveals, reward
    = top-1 accuracy minus patch_cost*K.
  - Agents A and D only (no dynamics model required). B and C need
    a forward-dynamics pretrain step which is a separate ~6 GPU-hour
    job; logging "pending" for now.
  - PPO via stable-baselines3, 50k env-steps per (agent, seed).
  - 2 agents x 3 seeds = 6 jobs in parallel via .map.

Outputs land on the cs224r-ckpts Modal Volume and are also returned
to the local driver for upload to github.
"""
from __future__ import annotations

import modal


app = modal.App("cs224r-real-imagenet")

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
    n_env_steps: int = 50_000,
    n_classes: int = 100,
    images_per_class_train: int = 50,
    images_per_class_val:   int = 10,
) -> dict:
    import json
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

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data --------------------------------------------------------
    ds_train = load_from_disk("/data/imagenet/train")
    ds_val   = load_from_disk("/data/imagenet/validation")
    print(f"loaded HF imagenet: train={len(ds_train)} val={len(ds_val)}")

    # Pick a deterministic 100-class subset
    rng_classes = np.random.default_rng(0)
    selected = sorted(rng_classes.choice(1000, size=n_classes, replace=False).tolist())
    cls_to_local = {c: i for i, c in enumerate(selected)}

    # Pull per-class image indices
    print(f"sampling {images_per_class_train} train / {images_per_class_val} val per class...")
    t0 = time.time()
    train_imgs = []  # (PIL, local_label)
    val_imgs   = []
    labels = np.array(ds_train["label"])
    for c in selected:
        idxs = np.where(labels == c)[0]
        np_rng.shuffle(idxs)
        for i in idxs[:images_per_class_train]:
            train_imgs.append((ds_train[int(i)]["image"], cls_to_local[c]))
    labels_v = np.array(ds_val["label"])
    for c in selected:
        idxs = np.where(labels_v == c)[0]
        np_rng.shuffle(idxs)
        for i in idxs[:images_per_class_val]:
            val_imgs.append((ds_val[int(i)]["image"], cls_to_local[c]))
    print(
        f"sampled train={len(train_imgs)} val={len(val_imgs)} in {time.time()-t0:.1f}s"
    )

    # ---- ViT-small backbone (frozen) --------------------------------
    print("loading timm vit_small_patch16_224 (pretrained)...")
    bb = timm.create_model("vit_small_patch16_224", pretrained=True,
                           num_classes=0).to(device).eval()
    for p in bb.parameters():
        p.requires_grad = False
    bb_feat_dim = bb.num_features

    # Fine-tune a fresh head on whole-image features for 1 epoch.
    head = torch.nn.Linear(bb_feat_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    from torchvision import transforms
    norm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def to_tensor(pil):
        return norm(pil.convert("RGB")).to(device)

    print("pretraining classifier head on whole-image features (1 epoch)...")
    head.train()
    rng_perm = np_rng.permutation(len(train_imgs))
    BATCH = 32
    for i in range(0, len(train_imgs), BATCH):
        batch_pil = [train_imgs[j] for j in rng_perm[i:i + BATCH]]
        x = torch.stack([to_tensor(p[0]) for p in batch_pil])
        y = torch.tensor([p[1] for p in batch_pil], device=device)
        with torch.no_grad():
            f = bb(x)
        logits = head(f)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    # Validate
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(val_imgs), BATCH):
            batch = val_imgs[i:i + BATCH]
            x = torch.stack([to_tensor(p[0]) for p in batch])
            y = torch.tensor([p[1] for p in batch], device=device)
            logits = head(bb(x))
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    bb_acc = correct / total
    print(f"backbone head val acc: {bb_acc:.3f}")

    # ---- foveated env ------------------------------------------------
    GRID = 7
    PATCH_SIZE = 224 // GRID  # 32
    N_PATCHES = GRID * GRID
    K = 8
    PATCH_COST = 0.01

    # Pre-compute per-patch features for the train + val sets so the
    # env can read them without re-running the backbone.
    def patch_features(images, label_pairs):
        feats_per_image = []
        labels_out = []
        bb.eval()
        with torch.no_grad():
            for pil, lbl in label_pairs:
                x = to_tensor(pil).unsqueeze(0)
                tokens = bb.forward_features(x)
                # tokens shape: (1, n_tokens+1, dim); drop CLS, reshape.
                # For ViT-S patch16, n_tokens = 196 = 14*14. We need
                # 7x7 = 49, so 2x2 pool.
                cls = tokens[:, 0, :]
                patch_tokens = tokens[:, 1:, :]  # (1, 196, 384)
                grid14 = patch_tokens.reshape(1, 14, 14, -1)
                # Pool to 7x7
                grid7 = grid14.reshape(1, 7, 2, 7, 2, -1).mean(dim=(2, 4))  # (1, 7, 7, 384)
                feats = grid7.reshape(N_PATCHES, -1)
                feats_per_image.append(feats.cpu().numpy())
                labels_out.append(lbl)
        return feats_per_image, labels_out

    print("caching per-patch features for train images...")
    train_feats, train_labels = patch_features(bb, train_imgs)
    print(f"cached {len(train_feats)} train tensors of shape "
          f"({N_PATCHES}, {train_feats[0].shape[1]})")
    print("caching per-patch features for val images...")
    val_feats, val_labels = patch_features(bb, val_imgs)
    feat_dim = train_feats[0].shape[1]

    class FoveatedImageNet(gym.Env):
        def __init__(self, feats, labels):
            super().__init__()
            self.feats = feats
            self.labels = labels
            obs_dim = N_PATCHES + N_PATCHES * 8 + N_PATCHES
            self.action_space = spaces.Discrete(N_PATCHES)
            self.observation_space = spaces.Box(
                low=-10, high=10, shape=(obs_dim,), dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            i = int(np.random.randint(len(self.feats)))
            self.f = self.feats[i]
            self.label = int(self.labels[i])
            self.mask = np.zeros(N_PATCHES, dtype=np.float32)
            self.k = 0
            # Reduced feature: per-patch summary = first 8 dims of feats
            self.f_red = self.f[:, :8].astype(np.float32)
            self.feat_buf = np.zeros((N_PATCHES, 8), dtype=np.float32)
            self.residual = np.zeros(N_PATCHES, dtype=np.float32)
            return self._obs(), {}

        def _obs(self):
            return np.concatenate([
                self.mask, self.feat_buf.reshape(-1), self.residual,
            ]).astype(np.float32)

        def _predict(self):
            mask_mat = self.mask[:, None]
            pooled = (self.feat_buf * mask_mat).sum(axis=0)
            pooled /= max(1.0, self.mask.sum())
            # Project back to 384-D via zero-pad
            z = np.zeros(feat_dim, dtype=np.float32)
            z[:8] = pooled
            with torch.no_grad():
                logits = head(torch.from_numpy(z).to(device))
            return int(logits.argmax().item())

        def step(self, action):
            action = int(action)
            terminated = truncated = False
            reward = 0.0
            if self.mask[action] == 1.0:
                terminated = True
            else:
                feat = self.f_red[action]
                pre_mean = self.feat_buf.sum(0) / max(1.0, self.mask.sum())
                residual = float(np.linalg.norm(feat - pre_mean) ** 2)
                self.residual[action] = residual
                self.feat_buf[action] = feat
                self.mask[action] = 1.0
                self.k += 1
                reward -= PATCH_COST
            if self.k >= K:
                truncated = True
                pred = self._predict()
                reward += 1.0 if pred == self.label else 0.0
            return self._obs(), reward, terminated, truncated, {
                "label": self.label,
                "pred": self._predict() if (terminated or truncated) else -1,
            }

    # Agent-specific wrappers
    if agent == "A":
        # baseline: zero out the residual channel so the policy sees
        # only the committed mask + feat buf
        class A_(FoveatedImageNet):
            def _obs(self):
                return np.concatenate([
                    self.mask, self.feat_buf.reshape(-1),
                    np.zeros(N_PATCHES, dtype=np.float32),
                ]).astype(np.float32)
        env_cls = A_
    elif agent == "D":
        # entropy-as-feature: residual channel replaced by classifier
        # entropy
        class D_(FoveatedImageNet):
            def _obs(self):
                mask_mat = self.mask[:, None]
                pooled = (self.feat_buf * mask_mat).sum(0)
                pooled /= max(1.0, self.mask.sum())
                z = np.zeros(feat_dim, dtype=np.float32)
                z[:8] = pooled
                with torch.no_grad():
                    p_logits = head(torch.from_numpy(z).to(device))
                    p_probs = torch.softmax(p_logits, dim=-1)
                    ent = float((-p_probs * torch.log(p_probs + 1e-9)).sum().item())
                ent_channel = np.full(N_PATCHES, ent, dtype=np.float32)
                return np.concatenate([
                    self.mask, self.feat_buf.reshape(-1), ent_channel,
                ]).astype(np.float32)
        env_cls = D_
    else:
        raise ValueError(f"agent {agent} requires a dynamics checkpoint; "
                         "out of scope for this entry point")

    env = DummyVecEnv([lambda: env_cls(train_feats, train_labels)])

    run = wandb.init(
        project="cs224r-foveated-imagenet",
        job_type="train",
        tags=["real", "imagenet100", f"agent_{agent}", f"seed_{seed}"],
        config={
            "seed": seed, "agent": agent, "n_classes": n_classes,
            "images_per_class_train": images_per_class_train,
            "n_env_steps": n_env_steps,
            "backbone_val_acc": bb_acc,
            "K": K, "GRID": GRID,
        },
    )

    class LogCb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rs, self.accs = [], []

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.rs.append(float(info["episode"]["r"]))
                if "label" in info and info.get("pred", -1) >= 0:
                    self.accs.append(1.0 if info["pred"] == info["label"] else 0.0)
            if len(self.rs) and len(self.rs) % 50 == 0:
                mean_r = float(np.mean(self.rs[-200:]))
                mean_a = float(np.mean(self.accs[-200:])) if self.accs else 0.0
                wandb.log({
                    "train/mean_reward": mean_r,
                    "train/mean_acc":    mean_a,
                    "train/n_episodes":  len(self.rs),
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
    print(f"trained in {train_time:.1f}s, episodes={len(cb.rs)}")

    # ---- eval on val set --------------------------------------------
    eval_env = env_cls(val_feats, val_labels)
    eval_rewards, eval_accs = [], []
    for _ in range(200):
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

    print(f"agent={agent} seed={seed}  eval acc={eval_mean_acc:.3f}  "
          f"reward={eval_mean_reward:.3f}")

    # Save the policy to the ckpt volume so this is reusable.
    ckpt_dir = Path(f"/ckpt/agent{agent}/seed{seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir / "ppo")
    ckpt_vol.commit()

    return {
        "agent": agent,
        "seed": seed,
        "n_env_steps": n_env_steps,
        "n_classes": n_classes,
        "backbone_val_acc": bb_acc,
        "train_time_s": train_time,
        "n_train_episodes": len(cb.rs),
        "eval_mean_reward": eval_mean_reward,
        "eval_mean_acc": eval_mean_acc,
        "wandb_url": run.url,
    }


@app.local_entrypoint()
def main(
    agents: str = "A,D",
    seeds: str  = "0,1,2",
    n_env_steps: int = 50_000,
    n_classes: int   = 100,
):
    import json
    from pathlib import Path

    agent_list = [a.strip() for a in agents.split(",") if a.strip()]
    seed_list  = [int(s) for s in seeds.split(",") if s.strip()]
    args = [(a, s, int(n_env_steps), int(n_classes), 50, 10)
            for a in agent_list for s in seed_list]
    print(f"launching {len(args)} parallel training jobs: {agent_list} x {seed_list}")
    out = list(train_one.starmap(args))

    Path("/tmp/real_cs224r_imagenet_result.json").write_text(json.dumps(out, indent=2))
    for o in out:
        print(
            f"  agent={o['agent']} seed={o['seed']}  "
            f"backbone={o['backbone_val_acc']:.3f}  "
            f"eval_acc={o['eval_mean_acc']:.3f}  "
            f"eps={o['n_train_episodes']}  "
            f"time={o['train_time_s']:.0f}s"
        )
