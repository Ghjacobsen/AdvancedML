# Part B Analysis: Sampling Quality of Generative Models

All experiments use standard (non-binarized) MNIST, 5000 FID samples, T=1000 diffusion steps.

---

## Task 1 — Show four representative samples from each model

**Status: Complete** → `reports/figures/ddpm_samples.png`, `latent_ddpm_samples.png`, `part_a_mog_samples_b.png`

Three sample grids were produced (4 images each):
- **DDPM**: Clean, sharp, clearly recognizable digits. The image-space diffusion model produces the highest visual quality.
- **Latent DDPM** (β=1): Noticeably blurrier and less structured than image-space DDPM. The VAE's Gaussian decoder introduces smoothing, and the latent DDPM at β=1 struggles to model the aggregate posterior accurately.
- **Part A VAE (MoG prior)**: Binary-looking samples (trained on binarized MNIST), structurally recognizable digits but lower fidelity than DDPM.

---

## Task 2 — Compute and report FID for all three models; for latent DDPM also report FID for different values of β including β=10⁻⁶

**Status: Complete** → `reports/figures/latent_ddpm_fid_vs_beta.png`, `reports/part_b_results.json`

### FID scores (main comparison)

| Model | FID |
|---|---|
| DDPM (image-space) | **5.10** |
| Part A VAE (MoG prior) | 18.25 |
| Latent DDPM (β=1) | 95.29 |

The image-space DDPM achieves by far the best FID at 5.10 — close to state-of-the-art for MNIST-scale models. The Part A MoG VAE scores 18.25, reasonable for a simple VAE. The Latent DDPM at β=1 performs poorly at 95.29.

### FID vs β for the Latent DDPM

| β | FID |
|---|---|
| 1×10⁻⁶ | 40.21 |
| 1×10⁻⁴ | 48.80 |
| 1×10⁻² | **39.28** |
| 1×10⁰ | 80.75 |

**Key finding:** Lower β substantially improves the latent DDPM's FID. At β=1 (standard VAE), the KL penalty strongly regularizes the encoder toward N(0,I), collapsing digit-specific structure into an undifferentiated Gaussian blob. The DDPM then has a near-Gaussian distribution to model — which is trivial but produces poor image quality because the decoder has little latent structure to decode. At small β (e.g. 10⁻²), the KL is weakly enforced, the aggregate posterior retains rich structure (spread out, class-separated clusters), and the latent DDPM can learn a more meaningful generative distribution, leading to lower FID. The optimal appears around β=10⁻², after which relaxing β further (10⁻⁶) gives slightly worse results, possibly due to near-deterministic encodings that are harder to sample from via diffusion.

---

## Task 3 — Measure and report wall-clock sampling time for all three models

**Status: Complete** → printed in job log and stored in `part_b_results.json`

| Model | Samples/second |
|---|---|
| Part A VAE (MoG) | 387,672 |
| Latent DDPM | 394 |
| DDPM (image-space) | 8.79 |

The VAE is roughly **44,000× faster** than the image-space DDPM. Sampling from a VAE is a single forward pass through the decoder. The Latent DDPM is intermediate — it runs T=1000 DDPM steps in the low-dimensional (20-dim) latent space, then one decoder pass. The image-space DDPM runs T=1000 steps of a U-Net over full 28×28 images, making it by far the slowest.

---

## Task 4 — Discuss and compare sampling quality and FID scores across the three models in relation to their sampling times

**Status: This is a report writing task — analysis below for use in the report**

There is a clear quality-vs-speed trade-off:

- **DDPM** achieves the best FID (5.10) but at only 8.79 samples/s. Every sample requires 1000 U-Net forward passes over the full image.
- **Latent DDPM** (best β=10⁻²: FID 39.28) is 45× faster than DDPM (394 vs 8.79 samples/s). The latent-space diffusion is cheap because the U-Net is replaced by a small MLP operating on 20-dimensional vectors. However, the Gaussian decoder introduces a quality ceiling — even with a perfect latent distribution, the MSE-trained decoder produces blurry images.
- **VAE (MoG)** is effectively instant (387,672 samples/s) but achieves only FID 18.25. It is faster than the latent DDPM by a factor of ~1000, and better FID too — suggesting that for simple datasets like MNIST the latent DDPM overhead is not justified unless β is tuned.

The Latent DDPM at β=1 is the worst of both worlds: slow (1000 latent diffusion steps) and poor quality (FID 95.29). Reducing β to 10⁻² recovers a more useful FID of 39.28, but still substantially worse than the direct image-space DDPM.

---

## Task 5 — Plot, discuss, and compare the β-VAE prior, the learned latent DDPM distribution, and the aggregate posterior

**Status: Complete** → `reports/figures/latent_distribution_comparison.png`

Three distributions are shown in 2D PCA space (the latent space is 20-dimensional):

- **VAE prior N(0, I)**: A round, isotropic Gaussian blob — by construction. At β=1 the aggregate posterior is strongly pulled toward this shape.
- **Latent DDPM distribution**: Samples from the trained DDPM operating in latent space. Ideally this should match the aggregate posterior. At β=1 the latent DDPM distribution is close to the prior (because the prior and posterior nearly coincide), which explains the poor FID — the DDPM learns to sample noise rather than meaningful latents.
- **Aggregate posterior q(z|x)**: The mean encodings of training images. These form class-separated clusters in PCA space, color-coded by digit label. The mismatch between this structured distribution and the isotropic prior (at β=1) is the root cause of the latent DDPM's poor performance.

**Discussion:** The comparison illustrates why β matters. At β=1 the posterior is squeezed into the prior, destroying cluster structure. At small β, the posterior preserves digit clusters, giving the latent DDPM a meaningful distribution to learn. However, the prior then becomes a poor approximation of the posterior, meaning naive N(0,I) sampling is less reliable — which motivates using the DDPM to learn the true posterior shape.

---

## Architecture descriptions (required in report)

### DDPM — U-Net

The image-space DDPM uses a U-Net with:
- **Input**: 1-channel 28×28 image
- **Time conditioning**: Sinusoidal timestep embedding → MLP → added to residual block activations
- **Encoder path**: Conv(1→C) → ResBlock(C→C) → Downsample → ResBlock(C→2C) → Downsample → (C=64, so 64→128 channels, 28→14→7 spatial)
- **Bottleneck**: ResBlock(2C→4C) → ResBlock(4C→4C) (256 channels at 7×7)
- **Decoder path**: Upsample → ResBlock(4C+2C→2C) → Upsample → ResBlock(2C+C→C) with skip connections
- **Output**: GroupNorm → SiLU → Conv(C→1)
- **Residual blocks**: GroupNorm(8) → SiLU → Conv3×3 → time projection → GroupNorm(8) → SiLU → Conv3×3 + skip

### Latent DDPM — MLP noise predictor

The latent DDPM operates on 20-dimensional latent codes from the β-VAE:
- **β-VAE encoder**: MLP with hidden_dim=256, outputs mean and log-std of q(z|x)
- **β-VAE decoder**: MLP with hidden_dim=256, maps z∈ℝ²⁰ → 28×28 image means (Gaussian likelihood, fixed σ=1)
- **Latent noise predictor**: cat(z_t, t_emb) → Linear(20+128, 512) → SiLU → Linear(512,512) → SiLU → Linear(512,512) → SiLU → Linear(512,20), where t_emb is a 128-dim sinusoidal time embedding
- **Sampling**: DDIM (deterministic, η=0) allowing variable step counts without retraining

---

## Summary: Is Part B complete?

| Requirement | Status |
|---|---|
| Train DDPM (U-Net) on standard MNIST | ✅ |
| Train Latent DDPM (β-VAE + latent diffusion) | ✅ |
| 4 samples from DDPM | ✅ |
| 4 samples from Latent DDPM | ✅ |
| 4 samples from a VAE of choice | ✅ (MoG VAE from Part A) |
| FID for all 3 models | ✅ |
| FID for latent DDPM across β values including β=10⁻⁶ | ✅ (β = 1e-6, 1e-4, 1e-2, 1.0) |
| Wall-clock sampling times for all 3 models | ✅ |
| Discuss quality vs speed trade-offs | ⚠️ Write in report |
| Plot prior vs latent DDPM vs aggregate posterior | ✅ |
| Discuss prior/DDPM/posterior comparison | ⚠️ Write in report |
| Briefly describe DDPM and latent DDPM architectures | ⚠️ Write in report |

The computational experiments are **fully complete**. The remaining work is writing the discussion paragraphs in the actual report PDF using the analysis above.
