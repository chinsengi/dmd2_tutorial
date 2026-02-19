# DMD2 Implementation in FastVideo

This document describes how Distribution Matching Distillation 2 (DMD2) is implemented in the FastVideo codebase, based on the paper [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867) (Yin et al., 2024). The goal is to distill a many-step diffusion model (teacher) into a few-step model (student) — typically 3 steps.

## DMD2 Paper Summary

### Background: Score Functions and Diffusion

A diffusion model learns a **score function** — given a noisy sample at some noise level, it predicts the direction to denoise toward clean data. At any timestep $t$, the score tells you "which way is the clean data from here." The full generative process chains many such small denoising steps from pure noise to a clean sample.

Distillation aims to collapse this many-step chain into a few steps (or even one). The challenge: a student that takes large steps will produce outputs from a *different distribution* than the teacher's real data distribution. You need a training signal that tells the student how its output distribution differs from the target.

### The Core Idea

DMD2 frames distillation as **distribution matching**. It uses two score functions to measure the gap between where the student's outputs land and where they should land:

- **Real score** $s_\text{real}(x_t, t)$: The score of the real data distribution. A frozen copy of the pretrained teacher model. Given a noisy sample, it points toward what real data looks like.
- **Fake score** $s_\text{fake}(x_t, t)$: The score of the generator's current output distribution. A separate trainable network. Given a noisy sample, it points toward what the generator's outputs look like.

Intuitively, the real data distribution matches the fake distribution produced by the student generator when the real and fake scores align. It turns out that the **difference** between these two scores at the student's output scales the gradient of the KL divergence between the student distribution and the real data distribution.

### Loss Functions in the Paper

The paper defines three loss components:

**1. Distribution Matching Loss (Eq. 2):**

The objective is to minimize the expected KL divergence between the generator's output distribution and the real data distribution across all noise levels $t$:

$$\mathcal{L}_{\text{DMD}} = \mathbb{E}_t \left[ \text{KL}(p_{\text{fake},t} \| p_{\text{real},t}) \right]$$

where $p_{\text{fake},t}$ is the marginal distribution of noised generator outputs and $p_{\text{real},t}$ is the marginal distribution of noised real data at level $t$.

#### Derivation: KL divergence → score difference gradient

**Step 1 — Expand the KL into entropy and cross-entropy:**

$$\text{KL}(p_{\text{fake},t} \| p_{\text{real},t}) = \underbrace{\mathbb{E}_{x_t \sim p_{\text{fake},t}} [\log p_{\text{fake},t}(x_t)]}_{\text{negative entropy}} - \underbrace{\mathbb{E}_{x_t \sim p_{\text{fake},t}} [\log p_{\text{real},t}(x_t)]}_{\text{negative cross-entropy}}$$

**Step 2 — Reparameterize using the forward diffusion process:**

The generator produces $x_0 = G_\theta(z)$ with $z \sim p(z)$. The forward diffusion adds noise:

$$x_t = F(G_\theta(z), t) = \alpha_t G_\theta(z) + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This reparameterization lets us move the $\theta$-gradient inside the expectation via the chain rule. For any function $f(x_t)$:

$$\nabla_\theta \, \mathbb{E}_{z, \epsilon}[f(x_t)] = \mathbb{E}_{z, \epsilon}\left[\nabla_{x_t} f(x_t) \cdot \frac{\partial x_t}{\partial \theta}\right] = \mathbb{E}_{z, \epsilon}\left[\nabla_{x_t} f(x_t) \cdot \alpha_t \nabla_\theta G_\theta(z)\right]$$

**Step 3 — Gradient of the negative entropy term:**

$$\nabla_\theta \, \mathbb{E}_{z, \epsilon}[\log p_{\text{fake},t}(x_t)] = \mathbb{E}_{z, \epsilon}\left[\nabla_{x_t} \log p_{\text{fake},t}(x_t) \cdot \alpha_t \nabla_\theta G_\theta(z)\right] = \mathbb{E}_{z, \epsilon}\left[s_\text{fake}(x_t, t) \cdot \alpha_t \nabla_\theta G_\theta(z)\right]$$

where $s_\text{fake}(x_t, t) = \nabla_{x_t} \log p_{\text{fake},t}(x_t)$ is the score of the generator's distribution.

**Step 4 — Gradient of the negative cross-entropy term:**

$$-\nabla_\theta \, \mathbb{E}_{z, \epsilon}[\log p_{\text{real},t}(x_t)] = -\mathbb{E}_{z, \epsilon}\left[\nabla_{x_t} \log p_{\text{real},t}(x_t) \cdot \alpha_t \nabla_\theta G_\theta(z)\right] = -\mathbb{E}_{z, \epsilon}\left[s_\text{real}(x_t, t) \cdot \alpha_t \nabla_\theta G_\theta(z)\right]$$

**Step 5 — Combine:**

$$\nabla_\theta \text{KL}(p_{\text{fake},t} \| p_{\text{real},t}) = \mathbb{E}_{z, \epsilon}\left[\left(s_\text{fake}(x_t, t) - s_\text{real}(x_t, t)\right) \cdot \alpha_t \nabla_\theta G_\theta(z)\right]$$

Averaging over $t$, this is the paper's Eq. 2:

$$\nabla_\theta \mathcal{L}_\text{DMD} = -\mathbb{E}_{t, z, \epsilon}\left[\left(s_\text{real}(x_t, t) - s_\text{fake}(x_t, t)\right) \cdot \alpha_t \nabla_\theta G_\theta(z)\right]$$

The negative sign reflects that we minimize the KL, so the generator is pushed in the direction where $s_\text{real} - s_\text{fake} > 0$ — i.e., where the real score "pulls harder" than the fake score.

> **Note:** The paper's Eq. 2 writes $\frac{dG_\theta(z)}{d\theta}$ without the $\alpha_t$ factor, even though the full derivation above shows it should be $\alpha_t \nabla_\theta G_\theta(z)$. The $\alpha_t$ comes from $\partial x_t / \partial G_\theta(z) = \alpha_t$ and is a scalar that scales the gradient magnitude per timestep. The paper absorbs it into the implicit weighting over the timestep distribution, treating it as part of $w(t)$ rather than writing it explicitly. In practice it doesn't affect the direction of optimization, only the effective per-timestep learning rate.

#### Connecting scores to model predictions

The scores are intractable but can be approximated from the diffusion model's denoising output. For a model predicting $\hat{x}_0(x_t, t)$ (the $x_0$-prediction / flow matching convention used by Wan):

$$s(x_t, t) = \nabla_{x_t} \log p_t(x_t) \approx \frac{\alpha_t \hat{x}_0(x_t, t) - x_t}{\sigma_t^2}$$

The score difference then becomes:

$$s_\text{fake} - s_\text{real} = \frac{\alpha_t(\hat{x}_{0,\text{fake}} - \hat{x}_{0,\text{real}})}{\sigma_t^2}$$

This is why FastVideo computes `grad` as the difference between two `pred_noise_to_pred_video` outputs — these are the $\hat{x}_0$ predictions from the fake and real score models.

#### From gradient to MSE pseudo-loss

Directly optimizing with the gradient expression requires computing $\nabla_\theta G_\theta(z)$ (the generator Jacobian). The trick used in practice — borrowed from SDS (Score Distillation Sampling) — is to cast it as an MSE loss with a **detached target**:

$$\mathcal{L}_\text{DMD} = \frac{1}{2} \left\| G_\theta(z) - \text{sg}\left[G_\theta(z) - w \cdot \text{grad}\right] \right\|^2$$

where $\text{sg}[\cdot]$ is stop-gradient and $w$ is a scalar weight. Differentiating this w.r.t. $\theta$:

$$\nabla_\theta \mathcal{L}_\text{DMD} = \nabla_\theta G_\theta(z) \cdot w \cdot \text{grad}$$

This is exactly $\nabla_\theta G_\theta(z) \cdot (s_\text{fake} - s_\text{real})$ (up to constants), matching the score difference gradient from Step 5. The MSE formulation lets standard backprop handle the $\nabla_\theta G_\theta$ term automatically. This is what FastVideo implements:

```python
dmd_loss = 0.5 * F.mse_loss(
    original_latent.float(),
    (original_latent.float() - grad.float()).detach()   # sg[G_θ(z) - grad]
)
```

#### Gradient normalization

FastVideo adds a normalization not in the original paper:

```python
grad = (faker_score_pred_video - real_score_pred_video) \
     / torch.abs(original_latent - real_score_pred_video).mean()
```

The denominator $|G_\theta(z) - \hat{x}_{0,\text{real}}|$ measures how far the generator output is from what the teacher would predict — i.e., the scale of the teacher's correction. Dividing by this makes the gradient magnitude adaptive: large corrections are scaled down, preventing gradient explosion when the generator is far from the teacher's manifold.

**2. GAN Loss (Eq. 4):**
A discriminator head $D$ is attached to the fake score model's bottleneck, classifying diffused real vs. generated samples:

$$\mathcal{L}_\text{GAN} = \mathbb{E}[\log D(F(x, t))] + \mathbb{E}[-\log(D(F(G_\theta(z), t)))]$$

This provides a per-sample adversarial signal that complements the distribution-level matching gradient.

**3. Fake Score Denoising Loss:**
Standard denoising score matching applied to generator outputs (not real data), so the fake score model tracks the generator's evolving distribution.

### Training Pipeline Overview

![DMD2 Training Pipeline](https://arxiv.org/html/2405.14867v2/x3.png)
*Figure 3 from the DMD2 paper: the generator (left) is optimized via the distribution matching gradient from the real and fake score models, plus the GAN loss from the discriminator head on the fake score model.*

### Two-Phase Training (Two Time-Scale Update Rule)

The paper uses a **5:1 ratio** of fake score updates to generator updates, inspired by Heusel et al.'s two time-scale GAN training:

**Every step — Fake score update:**
Train the fake score model with denoising score matching on generator outputs, plus the GAN discriminator classification loss. This must run frequently because the generator's distribution shifts with every weight update.

**Every 5 steps — Generator update:**
Minimize the distribution matching loss + GAN loss. The score difference provides the gradient direction; the GAN loss adds a per-sample quality signal.

The paper's Appendix C shows the 5:1 ratio is optimal — 10:1 overshoots and 1:1 is unstable.

### Multi-Step Distillation (Section 4.4)

Uses a fixed predetermined timestep schedule $\{t_1, t_2, \ldots, t_N\}$ identical at train and inference time. Inference chains denoising and re-noising:

$$x_0 \sim \mathcal{N}(0, I); \quad \hat{x}_{t_i} = G_\theta(x_{t_i}, t_i); \quad x_{t_{i+1}} = \alpha_{t_{i+1}} \hat{x}_{t_i} + \sigma_{t_{i+1}} \epsilon$$

The paper's SDXL 4-step schedule uses $\{999, 749, 499, 249\}$.

### Backward Simulation / Data-Free Mode (Section 4.5)

Addresses training-inference mismatch: during training, the student denoises from noisy *real images*, but at inference it denoises from its own previous outputs. The fix is to simulate the student's own multi-step inference trajectory during training, then apply the DMD loss at an intermediate point along that trajectory.

### Key Differences from DMD1

| Aspect | DMD1 | DMD2 |
|--------|------|------|
| Regression loss | Required (expensive paired dataset) | Removed entirely |
| Fake score stability | Single update per generator step | 5 updates per generator step |
| GAN integration | None | Discriminator head on fake score model |
| Multi-step support | One-step only | Supports 1-4+ steps |
| Training data | 12M precomputed (noise, clean) pairs | Only ~500k real images for GAN |
| Performance ceiling | Capped by teacher quality | Can exceed teacher quality |

---

## Paper vs. FastVideo: Key Differences

Before diving into the implementation details, here's a summary of where FastVideo's implementation diverges from the DMD2 paper:

| Aspect | DMD2 Paper | FastVideo Implementation |
|--------|-----------|--------------------------|
| **GAN loss** | Core component — discriminator head on fake score model provides per-sample adversarial signal | **Not implemented.** No discriminator, no GAN loss anywhere in the training code. Only the distribution matching loss and fake score denoising loss are used. |
| **Real score CFG** | Teacher is used as-is (or with standard CFG) | Explicit **classifier-free guidance** on the real score: runs conditional + unconditional forward passes and blends with `real_score_guidance_scale` (default 3.5). |
| **Gradient normalization** | Paper derives gradient from raw score difference | Implementation adds **normalization** |generator_output - real_score_pred|.mean()` to stabilize gradient magnitude, plus `nan_to_num` for numerical safety. |
| **Training data** | ~500k real images for GAN discriminator, no paired dataset | **Pre-computed synthetic latents** (600k teacher-generated VAE latents) in default mode. Data-free mode also supported. |
| **Domain** | Image generation (ImageNet, SDXL) | **Video generation** (Wan, Hunyuan, etc.) with sequence parallelism, VSA sparse attention. |
| **Denoising steps** | 4-step schedule $\{999, 749, 499, 249\}$ for SDXL | 3-step schedule $\{1000, 757, 522\}$ for Wan models. |
| **Flow matching** | Paper uses DDPM-style noise prediction ($\epsilon$-prediction) | Implementation uses **flow matching** formulation — target is $\epsilon - G(z)$ (the velocity/flow), consistent with Wan's flow matching scheduler. |
| **Sparse attention** | Not discussed | **Sparse-distill** strategy: jointly trains DMD + VSA so the distilled model also learns to work with sparse attention. |
| **MoE routing** | Not discussed | Supports **timestep-based expert routing** for models with two transformer experts (Wan2.2). |
| **EMA** | Not a focus of the paper | Full **EMA_FSDP** implementation for generator weight averaging, used for validation and separate checkpoint saving. |

The most significant divergence is the **absence of the GAN loss**. The paper presents this as a key contribution — the discriminator provides per-sample gradients that complement the distribution-level matching signal and are credited with enabling DMD2 to exceed teacher quality. FastVideo relies solely on the distribution matching gradient + fake score denoising, which is closer to DMD1's approach but with DMD2's other improvements (no regression loss, multi-step, two time-scale updates).

---

## FastVideo Implementation

### Key Files

| File | Role |
|------|------|
| `fastvideo/training/distillation_pipeline.py` | Base `DistillationPipeline` class — all core logic |
| `fastvideo/training/wan_distillation_pipeline.py` | `WanDistillationPipeline` — Wan-specific entry point |
| `fastvideo/training/wan_i2v_distillation_pipeline.py` | Wan Image-to-Video distillation variant |
| `fastvideo/fastvideo_args.py` | `TrainingArgs` dataclass with DMD-specific arguments |
| `fastvideo/configs/pipelines/wan.py` | `WanDMDConfig` — denoising step schedule |
| `fastvideo/training/training_utils.py` | Checkpoint save/load, EMA, `shift_timestep`, etc. |
| `fastvideo/models/utils.py` | `pred_noise_to_pred_video` — converts flow prediction to clean video |
| `examples/distill/` | Training launch scripts (Slurm) |

### Three-Model Setup

On initialization (`load_modules`, line 82), three transformer copies are loaded:

| Name | Variable | Trainable | Purpose |
|------|----------|-----------|---------|
| Generator | `self.transformer` | Yes | Produces video from noisy input in few steps |
| Fake score | `self.fake_score_transformer` | Yes | Learns score of generator's output distribution |
| Real score | `self.real_score_transformer` | No (frozen) | Provides score of real data distribution |

All three start from the same pretrained checkpoint. The real score model is immediately frozen:

```python
self.real_score_transformer.requires_grad_(False)
self.real_score_transformer.eval()
```

Each trainable model gets its own AdamW optimizer and LR scheduler. The fake score model can have a separate learning rate (`fake_score_learning_rate`).

#### MoE (Mixture of Experts) Support

For models like Wan2.2 with two transformer experts, there are `_2` variants of each model (`transformer_2`, `fake_score_transformer_2`, `real_score_transformer_2`). A `boundary_timestep` (derived from `boundary_ratio`) routes to the appropriate expert based on the noise level:

```python
# distillation_pipeline.py:540-565
def _get_real_score_transformer(self, timestep):
    if timestep.item() < self.boundary_timestep:
        return self.real_score_transformer_2   # low-noise expert
    else:
        return self.real_score_transformer     # high-noise expert
```

### Training Data: Two Modes

**Default mode (`simulate_generator_forward=False`):**
Uses a pre-computed synthetic dataset of VAE latents (e.g., `Wan-Syn_77x448x832_600k` — 600k latents generated offline by the teacher). At training time, these clean latents are loaded from the dataset as `vae_latent`. Fresh noise and timesteps are sampled every step — only the clean side of the pair is pre-cached, not the noise.

**Data-free mode (`simulate_generator_forward=True`):**
No pre-computed latents needed. The student simulates its own multi-step inference trajectory from pure noise, and the training loss is applied at a random exit point. This corresponds to DMD2 Section 4.5 and avoids training-inference mismatch. Batches only need text embeddings — latents are zero-filled placeholders.

### Denoising Step Schedule

The student learns to denoise at a fixed set of timesteps rather than arbitrary ones. This is configured via `dmd_denoising_steps` in the pipeline config:

```python
# configs/pipelines/wan.py
class WanDMDConfig(WanT2V_1_3B_Config):
    dmd_denoising_steps = [1000, 757, 522]   # 3-step inference
```

These are loaded into `self.denoising_step_list` at initialization (line 250). When `warp_denoising_step=True`, they're remapped through the scheduler's shifted timestep schedule.

### Training Loop

`train_one_step` (line 959) alternates between two phases, implementing the DMD2 two-phase training described above:

#### Phase 1: Generator Update (every N steps)

Controlled by `generator_update_interval` (default 5). Only runs when `current_trainstep % generator_update_interval == 0`.

**Step 1 — Generator forward** (`_generator_forward`, line 585):
- Pick a random timestep from `denoising_step_list`
- Add freshly sampled noise to the ground-truth latent at that timestep
- Run the student transformer to predict flow, convert to predicted clean video via `pred_noise_to_pred_video`

**Step 2 — DMD loss** (`_dmd_forward`, line 704):

All score computations happen under `torch.no_grad()` — gradients only flow through the final MSE back to the generator.

1. Sample a random timestep $t$, clamp to `[min_timestep, max_timestep]`, and add noise to the generator output:
   ```python
   noisy_latent = scheduler.add_noise(generator_pred_video, noise, t)
   ```

2. Run the **fake score** model on this noisy input to get its predicted clean video:
   ```python
   fake_score_pred_noise = fake_score_transformer(noisy_latent, t, text_emb)
   faker_score_pred_video = pred_noise_to_pred_video(fake_score_pred_noise, ...)
   ```

3. Run the **real score** model twice — conditional and unconditional — and apply classifier-free guidance:
   ```python
   real_score_pred_video = pred_cond + guidance_scale * (pred_cond - pred_uncond)
   ```

4. Compute the **distribution matching gradient** — the normalized difference between the two scores' predicted clean outputs:
   ```python
   grad = (faker_score_pred_video - real_score_pred_video)
        / |original_latent - real_score_pred_video|.mean()
   grad = torch.nan_to_num(grad)
   ```
   The denominator normalizes by how far the generator output is from the teacher's prediction, stabilizing the gradient magnitude.

5. Convert to an **MSE loss** that pushes the generator in the gradient direction:
   ```python
   dmd_loss = 0.5 * F.mse_loss(
       original_latent.float(),
       (original_latent.float() - grad.float()).detach()
   )
   ```
   The `.detach()` on the target means gradients only flow through `original_latent` (the generator output), not through the score computation. The generator learns to move its output in the direction that closes the gap between fake and real scores.

**Step 3** — Backprop through the generator, clip gradients, optimizer step, update EMA.

#### Phase 2: Fake Score Update (every step)

Runs unconditionally every training step, keeping the fake score model synchronized with the generator's evolving output distribution.

**Fake score forward** (`faker_score_forward`, line 804):

1. Generate a prediction from the student (no grad) — using either single-step or multi-step simulation depending on `simulate_generator_forward`.

2. Add noise at a random timestep to the generator's output:
   ```python
   noisy_generator_pred_video = scheduler.add_noise(generator_pred_video, noise, t)
   ```

3. Run the fake score model to predict noise:
   ```python
   fake_score_pred_noise = fake_score_transformer(noisy_generator_pred_video, t, text_emb)
   ```

4. The target is the **flow** from the generator output to the noise:
   ```python
   target = fake_score_noise - generator_pred_video
   ```

5. Standard flow matching MSE loss:
   ```python
   flow_matching_loss = torch.mean((fake_score_pred_noise - target)**2)
   ```

This is a denoising score matching objective applied to the generator's output distribution rather than real data. It ensures the fake score model accurately tracks where the generator's outputs are, so the DMD gradient remains meaningful.

**Then** — Backprop through the fake score model, clip gradients, optimizer step.

#### Why the Fake Score Updates More Frequently

The generator updates every 5 steps; the fake score updates every step. This asymmetry is deliberate: each generator update shifts the output distribution, and the fake score needs several updates to re-converge to the new distribution before the next generator update can compute an accurate distribution matching gradient. If the fake score is stale, the DMD gradient would point in the wrong direction.

#### Gradient Accumulation

Both phases loop over `gradient_accumulation_steps` batches, dividing each loss by the accumulation count before `.backward()`. Losses are averaged across all ranks via `all_reduce(ReduceOp.AVG)`.

### Multi-Step Simulation Mode

When `simulate_generator_forward=True`, the generator forward pass (`_generator_multi_step_simulation_forward`, line 619) simulates the actual multi-step inference trajectory:

1. Start from pure noise
2. Run the student through intermediate denoising steps (no grad) to build up the trajectory
3. Apply the training loss only at a randomly chosen exit point

This gives a better training signal since it matches what happens at inference time, but is more expensive.

In this mode, the data batch doesn't need pre-computed VAE latents — it generates zero-filled latent placeholders instead (line 933).

### Integration with Training Infrastructure

#### FSDP2

All transformer models get activation checkpointing applied (`apply_activation_checkpointing`) for memory efficiency. Distributed communication uses `get_world_group().all_reduce()` for loss synchronization.

#### Checkpointing

`save_distillation_checkpoint` / `load_distillation_checkpoint` (in `training_utils.py`) persist all six models (generator, fake score, real score, plus their `_2` variants), both optimizers, both LR schedulers, the EMA state, the dataloader state, and the noise RNG.

#### EMA

When `ema_decay > 0`, an `EMA_FSDP` wrapper tracks exponential moving averages of the generator weights. EMA weights are used for validation and saved as separate checkpoints.

#### VSA (Video Sparse Attention) Integration

The training loop maintains two sets of attention metadata:
- `attn_metadata_vsa` — with VSA sparsity (used for generator forward, where sparsity speeds up inference)
- `attn_metadata` — with `VSA_sparsity=0.0` (used for score model forward, where full attention is needed for accurate scores)

This is the "Sparse-distill" strategy described in the docs — jointly training DMD + VSA so the distilled model also learns to work with sparse attention.

### Configuration Reference

Key `TrainingArgs` fields for DMD:

| Argument | Default | Description |
|----------|---------|-------------|
| `real_score_model_path` | `""` | Path to teacher model |
| `fake_score_model_path` | `""` | Path to initial fake score model |
| `generator_update_interval` | `5` | Update generator every N fake score steps |
| `real_score_guidance_scale` | `3.5` | CFG scale for teacher |
| `fake_score_learning_rate` | `0.0` | Separate LR for fake score (0 = use default) |
| `fake_score_lr_scheduler` | `"constant"` | LR schedule for fake score |
| `fake_score_betas` | `"0.9,0.999"` | AdamW betas for fake score |
| `min_timestep_ratio` | `0.2` | Min timestep as fraction of total |
| `max_timestep_ratio` | `0.98` | Max timestep as fraction of total |
| `simulate_generator_forward` | `False` | Use multi-step simulation |
| `warp_denoising_step` | `False` | Remap steps through scheduler shift |
| `ema_decay` | `None` | EMA decay rate (0 = disabled) |
| `boundary_ratio` | `None` | MoE expert routing threshold |

Pipeline config: `dmd_denoising_steps` (e.g., `[1000, 757, 522]` for 3-step inference).

### Launching Training

Training is launched via `torchrun` pointing at the model-specific pipeline script:

```bash
torchrun --nnodes $NNODES --nproc_per_node 8 \
    fastvideo/training/wan_distillation_pipeline.py \
    --model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --real_score_model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --fake_score_model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --dmd_denoising_steps '1000,757,522' \
    --generator_update_interval 5 \
    --real_score_guidance_scale 3.5 \
    --learning_rate 1e-5 \
    --max_train_steps 4000
```

Example Slurm scripts are in `examples/distill/Wan2.1-T2V/`.
