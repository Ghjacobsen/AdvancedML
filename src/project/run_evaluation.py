"""
Post-training evaluation script for Part A: Priors for VAEs.

Loads all trained checkpoints (3 priors × 3 runs) and produces:
1. Per-datapoint test ELBO mean ± std across runs for each prior
2. Prior vs aggregate posterior plots (best run per prior)
3. Generated samples grid (best run per prior)
4. Reconstruction comparison (best run per prior)
5. ELBO comparison bar chart
6. JSON results summary

Usage (after training jobs finish):
    uv run python src/project/run_evaluation.py
    uv run python src/project/run_evaluation.py --models-dir models --output-dir reports
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from project.data import get_mnist_loaders, get_full_test_loader
from project.evaluate import compute_test_elbo, load_model
from project.visualize import (
    plot_prior_posterior,
    plot_samples,
    plot_reconstructions,
    plot_training_curves,
    plot_elbo_comparison,
)


PRIORS = ["gaussian", "mog", "flow"]
N_RUNS = 3
PRIOR_DISPLAY = {
    "gaussian": "Standard Gaussian",
    "mog": "Mixture of Gaussians",
    "flow": "Normalizing Flow",
}


def find_checkpoints(models_dir: str) -> Dict[str, List[Path]]:
    """Discover all trained checkpoint files grouped by prior type."""
    models_dir = Path(models_dir)
    checkpoints = {}
    for prior in PRIORS:
        ckpts = []
        for run in range(N_RUNS):
            # Prefer best checkpoint, fall back to final
            best = models_dir / prior / f"run_{run}" / f"vae_{prior}_best.pt"
            final = models_dir / prior / f"run_{run}" / f"vae_{prior}_final.pt"
            if best.exists():
                ckpts.append(best)
            elif final.exists():
                ckpts.append(final)
        if ckpts:
            checkpoints[prior] = ckpts
    return checkpoints


def evaluate_checkpoints(
    checkpoints: Dict[str, List[Path]],
    test_loader,
    device: torch.device,
) -> Dict[str, dict]:
    """Load each checkpoint and compute per-datapoint test ELBO."""
    results = {}
    for prior, ckpt_paths in checkpoints.items():
        elbos = []
        print(f"\n{'='*60}")
        print(f"Evaluating {PRIOR_DISPLAY[prior]} prior ({len(ckpt_paths)} runs)")
        print(f"{'='*60}")
        for i, path in enumerate(ckpt_paths):
            print(f"  Run {i}: {path.name} ... ", end="", flush=True)
            model, config = load_model(str(path), device)
            elbo = compute_test_elbo(model, test_loader, device)
            elbos.append(elbo)
            print(f"ELBO = {elbo:.2f}")

        results[prior] = {
            "elbos": elbos,
            "mean": float(np.mean(elbos)),
            "std": float(np.std(elbos)),
            "best_idx": int(np.argmax(elbos)),
            "best_path": str(ckpt_paths[int(np.argmax(elbos))]),
        }
    return results


def generate_plots(
    results: Dict[str, dict],
    train_loader,
    device: torch.device,
    figures_dir: Path,
):
    """Generate all required plots for the report."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    for prior, res in results.items():
        display = PRIOR_DISPLAY[prior]
        print(f"\n--- Generating plots for {display} ---")

        # Load the best model for this prior
        model, config = load_model(res["best_path"], device)

        # 1. Prior vs aggregate posterior
        plot_prior_posterior(
            model,
            train_loader,
            device,
            save_path=str(figures_dir / f"{prior}_prior_posterior.png"),
            n_samples=5000,
            prior_name=display,
        )

        # 2. Generated samples
        plot_samples(
            model,
            device,
            n_samples=64,
            save_path=str(figures_dir / f"{prior}_samples.png"),
            title=f"Generated Samples ({display} Prior)",
        )

        # 3. Reconstructions
        plot_reconstructions(
            model,
            train_loader,
            device,
            n_samples=10,
            save_path=str(figures_dir / f"{prior}_reconstructions.png"),
            title=f"Reconstructions ({display} Prior)",
        )

        # 4. Training curves (from final checkpoint if available)
        final_path = Path(res["best_path"]).parent / f"vae_{prior}_final.pt"
        if final_path.exists():
            ckpt = torch.load(str(final_path), map_location=device, weights_only=False)
            if "history" in ckpt:
                plot_training_curves(
                    ckpt["history"],
                    save_path=str(figures_dir / f"{prior}_training.png"),
                    title=f"Training Curves ({display} Prior)",
                )

    # 5. ELBO comparison bar chart
    comparison_data = {
        prior: {"mean": res["mean"], "std": res["std"], "all_runs": res["elbos"]}
        for prior, res in results.items()
    }
    plot_elbo_comparison(comparison_data, save_path=str(figures_dir / "elbo_comparison.png"))


def print_summary(results: Dict[str, dict]):
    """Print the final results table to stdout."""
    print("\n" + "=" * 70)
    print("PART A — TEST SET LOG-LIKELIHOOD (ELBO) COMPARISON")
    print("=" * 70)
    print(f"{'Prior':<25} {'Mean ELBO':<15} {'Std':<10} {'Runs'}")
    print("-" * 70)
    for prior in PRIORS:
        if prior not in results:
            continue
        res = results[prior]
        runs_str = ", ".join(f"{e:.2f}" for e in res["elbos"])
        print(f"{PRIOR_DISPLAY[prior]:<25} {res['mean']:<15.2f} {res['std']:<10.2f} [{runs_str}]")
    print("=" * 70)

    print("\nArchitecture summary:")
    print("  Encoder: Flatten(28×28) → Linear(784,256) → ReLU → Linear(256,256) → ReLU → Linear(256,2M)")
    print("  Decoder: Linear(M,256) → ReLU → Linear(256,256) → ReLU → Linear(256,784) → Unflatten(28,28)")
    print("  Latent dim M = 20, Bernoulli likelihood, Adam optimizer (lr=1e-3)")
    print("  Priors: Standard Gaussian N(0,I) | MoG (K=10 learnable components) | Flow (4 RealNVP coupling layers)")


def save_results(results: Dict[str, dict], output_path: Path):
    """Save results to JSON for reproducibility."""
    serializable = {
        "timestamp": datetime.now().isoformat(),
        "results": {
            prior: {
                "display_name": PRIOR_DISPLAY[prior],
                "mean_elbo": res["mean"],
                "std_elbo": res["std"],
                "per_run_elbos": res["elbos"],
                "best_run_idx": res["best_idx"],
                "best_checkpoint": res["best_path"],
            }
            for prior, res in results.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained VAE models for Part A")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Discover checkpoints
    checkpoints = find_checkpoints(args.models_dir)
    if not checkpoints:
        print("ERROR: No checkpoints found. Run training first.")
        return
    for prior, paths in checkpoints.items():
        print(f"  {prior}: {len(paths)} runs found")

    # 2. Load data
    train_loader, _ = get_mnist_loaders(batch_size=128, data_dir=args.data_dir)
    test_loader = get_full_test_loader(batch_size=256, data_dir=args.data_dir)

    # 3. Compute test ELBO per checkpoint
    results = evaluate_checkpoints(checkpoints, test_loader, device)

    # 4. Generate all plots
    figures_dir = Path(args.output_dir) / "figures"
    generate_plots(results, train_loader, device, figures_dir)

    # 5. Print summary table
    print_summary(results)

    # 6. Save results JSON
    save_results(results, Path(args.output_dir) / "part_a_results.json")


if __name__ == "__main__":
    main()
