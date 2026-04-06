# Design Decisions

Running log of open implementation questions. For each question: the default (what the phase docs currently assume), the decision, and the date + reasoning when it locks.

Edit this file as decisions are made. If a decision changes a phase doc materially, note which file needs updating.

---

## Q1 - Goal encoding at evaluation time

**Phase doc:** [phase2_ppo_baseline.md](phase2_ppo_baseline.md) §5  
**Blocks:** Phase 2, Step 2.2

**My default:** Goal is provided as a one-hot during *training only*. At eval on held-out corruptions, the goal channel is a zero vector - trained against via 10% random-dropout during training so the policy has a calibrated "unknown" behavior. This means "adaptation" = what the policy does with no regime information, relying only on its observations.

**Question:** Should the goal (corruption type) also be available at eval time on held-out corruptions? If yes, "adaptation" means something different - the policy is told the regime but hasn't practiced that regime.

**Your decision:**

**Date:**

**Reasoning:**

---

## Q2 - Dynamics model operating space

**Phase doc:** [phase3_dynamics_model.md](phase3_dynamics_model.md) §1  
**Blocks:** Phase 3

**My default:** Predict macro-patch ViT token embeddings (49 patches × 384 dims). Specifically, given the current low-res embedding + action + mask, predict what the ViT token embedding at that patch index would be if revealed.

**Question:** Should the dynamics model operate in raw ViT token space, a smaller learned projection of it, or something else? Mouhssine has context from the toy version.

**Your decision:**

**Date:**

**Reasoning:**

---

## Q3 - Precision weighting form

**Phase doc:** [phase3_dynamics_model.md](phase3_dynamics_model.md) §3  
**Blocks:** Phase 3

**My default:** Per-dimension running variance (one scalar per of the 384 embedding dims), updated via Welford's online algorithm during dynamics pretraining, then frozen. Residual fed to policy = `r / sqrt(σ² + ε)`, a 384-dim vector per patch.

**v1 simplification:** Per-patch residual *norm* (a 49-dim vector, one scalar per patch) instead of the full 384-dim vector. Reduces obs dimensionality and is easier to debug; full residual is an ablation.

**Question:** Is per-dim variance the right precision definition, or do you want something else (e.g., per-patch scalar, learned precision, global normalization)? What does the toy work use?

**Your decision:**

**Date:**

**Reasoning:**

---

## Q4 - Patch grid resolution

**Phase doc:** [phase1_environment.md](phase1_environment.md) §1  
**Blocks:** Phase 1, Step 1.4

**My default:** 7×7 = 49 patches, each 32×32 px, 8 reveals per episode. This keeps the action space tractable for PPO at the cost of not aligning natively with ViT-small's 14×14 patch grid.

**Question:** Do you have a prior from the toy work on patch grid size? 14×14 (native ViT alignment, 196 actions) vs. 7×7 (manageable PPO action space).

**Your decision:**

**Date:**

**Reasoning:**

---

## Q5 - Adaptation curve semantics

**Phase doc:** [phase5_evaluation.md](phase5_evaluation.md) §3.3  
**Blocks:** Phase 5 only

**My default:** Zero-shot streaming - no test-time gradient updates. Figure 2 shows cumulative mean accuracy over an increasing prefix of held-out-corruption episodes. "Adaptation" = the policy's zero-shot generalization, not online learning.

**Question:** Did you intend online test-time gradient updates (requires a meta-RL or fine-tuning loop at test time, not just streaming evaluation)?

**Your decision:**

**Date:**

**Reasoning:**

---

## Q6 - Held-out corruption split

**Phase doc:** [phase1_environment.md](phase1_environment.md) §3 step 1.2  
**Blocks:** Phase 1, Step 1.2

**My default:**
- **Training (10):** gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog
- **Held-out (5):** brightness, contrast, elastic_transform, pixelate, jpeg_compression
- **Severities trained:** 1-3. Severities 4-5 held out for stress eval.

This is the standard "noise+blur+weather → digital" split common in ImageNet-C papers.

**Question:** Is there a canonical split from the toy / lab work I should reuse instead?

**Your decision:**

**Date:**

**Reasoning:**

---

## Q7 - Dynamics model training schedule

**Phase doc:** [phase3_dynamics_model.md](phase3_dynamics_model.md) §8  
**Blocks:** Phase 3

**My default:** Pretrain the dynamics model offline on a fixed rollout dataset (mixed random-policy + 50%-trained Agent A), then **freeze** during PPO training for Agents B and C. This decouples the prediction-error signal from the RL optimizer and means the same dynamics checkpoint is shared across all seeds.

**Alternative:** Train dynamics jointly with PPO (one dynamics-loss step per rollout batch). More adaptive but introduces a non-stationarity in Agent C's feature distribution.

**Question:** Which schedule matches what you had in mind? Any strong preference from the toy work?

**Your decision:**

**Date:**

**Reasoning:**

---

## Q8 - Agent B intrinsic weight tuning budget

**Phase doc:** [phase4_agents_bc.md](phase4_agents_bc.md) §5  
**Blocks:** Phase 4, Step 4.2

**My default:** Small sweep - β ∈ {0.01, 0.1, 1.0, 5.0}, one seed, one training corruption, ~1M steps each. Pick the winner; use it for all 3 seeds. ~4 GPU-hours.

**Minimum viable alternative:** Use Pathak et al.'s reported β=0.01 (scaled to our reward range) without sweeping. ~0 extra GPU-hours.

**Question:** Is there budget and interest in the sweep, or should I just use the canonical ICM default?

**Your decision:**

**Date:**

**Reasoning:**

---

## Q9 - Classifier head: frozen pretrained vs. fine-tuned on foveated views

**Phase doc:** [phase1_environment.md](phase1_environment.md) §3 step 1.3  
**Blocks:** Phase 1, Step 1.3

**My default:** Fine-tune a fresh 2-layer MLP head (frozen backbone) on synthetic foveated views (random patch reveal counts). One pass over ImageNet train, saved to `/ckpt/heads/foveated_head_clean.pt`. Then frozen during RL.

**Rationale:** Pretrained ViT-small's classification head was trained on full-resolution inputs. On partially foveated images it systematically underperforms, which depresses reward and slows learning. A foveated-view-trained head gives the agent a higher reward ceiling to learn toward.

**Question:** Any concern with this approach? Is the off-the-shelf head worth trying first to avoid the extra pretraining step?

**Your decision:**

**Date:**

**Reasoning:**

---

## Q10 - Compute budget

**Phase doc:** Affects all phases  
**Blocks:** seed count decisions in Phases 2, 3, 4, 5

**My estimate:**
- Agent A, 3 seeds: ~15 GPU-hours
- Dynamics pretraining: ~3 GPU-hours
- Agents B + C, 3 seeds each: ~30 GPU-hours
- Evaluation sweep (all agents × corruptions): ~5 GPU-hours
- Agent B β-sweep: ~4 GPU-hours
- Ablations (dynamics size, etc.): ~15 GPU-hours
- **Total: ~70-120 GPU-hours on Modal (A10G)**

At Modal's current A10G spot pricing (~$0.76/hr), that's roughly $50-$90.

**Question:** Is this budget acceptable? If there's a hard cap, I'll reduce to 1 seed (halves training cost) and skip ablations.

**Your decision:**

**Date:**

**Reasoning:**
