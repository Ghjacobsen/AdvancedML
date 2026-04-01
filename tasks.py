import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "project"
PYTHON_VERSION = "3.12"

# =============================================================================
# Part A: VAE Experiments
# =============================================================================

@task
def train_vae(ctx: Context, prior: str = "gaussian", epochs: int = 50, seed: int = 42) -> None:
    """Train VAE with specified prior (gaussian, mog, flow)."""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/train.py --prior {prior} --epochs {epochs} --seed {seed}",
        echo=True, pty=not WINDOWS
    )

@task
def run_experiments(ctx: Context, quick: bool = False) -> None:
    """Run all Part A experiments (all priors, multiple runs)."""
    quick_flag = "--quick" if quick else ""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/run_experiments.py {quick_flag}",
        echo=True, pty=not WINDOWS
    )

@task
def evaluate_vae(ctx: Context, checkpoint: str = None, prior: str = None) -> None:
    """Evaluate VAE model on test set."""
    if checkpoint:
        ctx.run(
            f"uv run python src/{PROJECT_NAME}/evaluate.py --checkpoint {checkpoint}",
            echo=True, pty=not WINDOWS
        )
    elif prior:
        ctx.run(
            f"uv run python src/{PROJECT_NAME}/evaluate.py --prior {prior}",
            echo=True, pty=not WINDOWS
        )
    else:
        ctx.run(
            f"uv run python src/{PROJECT_NAME}/evaluate.py --prior all",
            echo=True, pty=not WINDOWS
        )

# =============================================================================
# Part B: Sampling Quality Experiments
# =============================================================================

@task
def run_part_b(ctx: Context, quick: bool = False, skip_training: bool = False,
               part_a_prior: str = "mog", ddpm_epochs: int = 100,
               vae_epochs: int = 50, latent_ddpm_epochs: int = 100) -> None:
    """Run Part B experiments: DDPM, Latent DDPM, FID, sampling times."""
    quick_flag = "--quick" if quick else ""
    skip_flag = "--skip-training" if skip_training else ""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/run_part_b.py "
        f"{quick_flag} {skip_flag} "
        f"--part-a-prior {part_a_prior} "
        f"--ddpm-epochs {ddpm_epochs} "
        f"--vae-epochs {vae_epochs} "
        f"--latent-ddpm-epochs {latent_ddpm_epochs}",
        echo=True, pty=not WINDOWS
    )

@task
def train_ddpm(ctx: Context, epochs: int = 100, base_channels: int = 64, T: int = 1000) -> None:
    """Train image-space DDPM on standard MNIST."""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/ddpm.py "
        f"--epochs {epochs} --base-channels {base_channels} --T {T}",
        echo=True, pty=not WINDOWS
    )

# =============================================================================
# Data
# =============================================================================

@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data (downloads MNIST)."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)

# =============================================================================
# Training (legacy)
# =============================================================================

@task
def train(ctx: Context) -> None:
    """Train model (legacy command)."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

# =============================================================================
# Testing
# =============================================================================

# =============================================================================
# Mini-project 2: Ensemble VAE Geometry
# =============================================================================

@task
def ensemble_train(ctx: Context, num_reruns: int = 10, num_decoders: int = 3,
                   epochs: int = 50, device: str = "cpu") -> None:
    """Train all ensemble VAEs (M retrainings × K decoders)."""
    ctx.run(
        f"PYTHONPATH=src:$PYTHONPATH uv run python src/project2/experiment.py train "
        f"--num-reruns {num_reruns} --num-decoders {num_decoders} "
        f"--epochs {epochs} --device {device}",
        echo=True, pty=not WINDOWS
    )

@task
def ensemble_geodesics(ctx: Context, num_curves: int = 25, retrain_idx: int = 0,
                       device: str = "cpu") -> None:
    """Compute geodesics for Part A and Part B."""
    ctx.run(
        f"PYTHONPATH=src:$PYTHONPATH uv run python src/project2/experiment.py geodesics "
        f"--num-curves {num_curves} --retrain-idx {retrain_idx} --device {device} --verbose",
        echo=True, pty=not WINDOWS
    )

@task
def ensemble_cov(ctx: Context, num_reruns: int = 10, num_cov_pairs: int = 10,
                 device: str = "cpu") -> None:
    """Compute CoV analysis across retrainings."""
    ctx.run(
        f"PYTHONPATH=src:$PYTHONPATH uv run python src/project2/experiment.py cov "
        f"--num-reruns {num_reruns} --num-cov-pairs {num_cov_pairs} --device {device} --verbose",
        echo=True, pty=not WINDOWS
    )

@task
def ensemble_plot(ctx: Context) -> None:
    """Re-generate all plots from saved results (no recomputing)."""
    ctx.run(
        "PYTHONPATH=src:$PYTHONPATH uv run python src/project2/experiment.py plot",
        echo=True, pty=not WINDOWS
    )

@task
def ensemble_all(ctx: Context, device: str = "cpu") -> None:
    """Run full ensemble pipeline: train → geodesics → CoV → plots."""
    ctx.run(
        f"PYTHONPATH=src:$PYTHONPATH uv run python src/project2/experiment.py all --device {device} --verbose",
        echo=True, pty=not WINDOWS
    )

# =============================================================================
# Testing
# =============================================================================

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

# =============================================================================
# Docker
# =============================================================================

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# =============================================================================
# Documentation
# =============================================================================

@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
