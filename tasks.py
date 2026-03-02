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
