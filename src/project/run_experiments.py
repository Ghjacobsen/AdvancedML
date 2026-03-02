"""
Main experiment runner for Part A: Priors for VAEs.

This script orchestrates the complete pipeline:
1. Train VAEs with all three priors (Gaussian, MoG, Flow)
2. Run multiple training runs for each prior
3. Generate prior vs posterior plots
4. Compute test ELBO with mean ± std
5. Generate comparison plots and summary

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --prior gaussian  # Run single prior
    python run_experiments.py --quick           # Quick test (1 run, 5 epochs)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

from project.data import get_mnist_loaders
from project.train import train_model, create_vae
from project.evaluate import compute_test_elbo, evaluate_multiple_runs, load_model
from project.visualize import (
    plot_prior_posterior,
    plot_samples,
    plot_reconstructions,
    plot_training_curves,
    plot_elbo_comparison,
    create_all_plots_for_prior,
)


def run_part_a_experiments(
    n_runs: int = 3,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 20,
    hidden_dim: int = 256,
    output_dir: str = "./reports",
    data_dir: str = "./data",
    models_dir: str = "./models",
    priors: Optional[list] = None,
    device: Optional[torch.device] = None,
):
    """
    Run complete Part A experiments.
    
    Parameters:
    n_runs: [int] Number of training runs per prior (for mean/std)
    epochs: [int] Training epochs
    batch_size: [int] Batch size
    lr: [float] Learning rate
    latent_dim: [int] Latent space dimension
    hidden_dim: [int] Hidden layer dimension
    output_dir: [str] Output directory for reports
    data_dir: [str] Directory for MNIST data
    models_dir: [str] Directory for model checkpoints
    priors: [list] List of priors to evaluate (default: all)
    device: [torch.device] Device to use
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if priors is None:
        priors = ["gaussian", "mog", "flow"]
    
    # Create directories
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PART A: VAE PRIORS EXPERIMENT")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Priors: {priors}")
    print(f"Runs per prior: {n_runs}")
    print(f"Epochs per run: {epochs}")
    print(f"Latent dim: {latent_dim}, Hidden dim: {hidden_dim}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    print("=" * 70)
    
    # Store results
    all_results = {}
    
    # Get data loader for visualization (shared)
    train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
    )
    
    for prior_type in priors:
        print(f"\n{'='*70}")
        print(f"PRIOR: {prior_type.upper()}")
        print(f"{'='*70}")
        
        prior_results = {
            "elbos": [],
            "best_elbo": float("-inf"),
            "best_model_path": None,
            "histories": [],
        }
        
        for run in range(n_runs):
            print(f"\n--- Run {run + 1}/{n_runs} ---")
            seed = 42 + run
            run_dir = f"{models_dir}/{prior_type}/run_{run}"
            
            # Train model
            model, history = train_model(
                prior_type=prior_type,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                seed=seed,
                save_dir=run_dir,
                data_dir=data_dir,
                device=device,
            )
            
            # Compute test ELBO
            test_elbo = compute_test_elbo(model, test_loader, device)
            prior_results["elbos"].append(test_elbo)
            prior_results["histories"].append(history)
            
            print(f"Run {run + 1} Test ELBO: {test_elbo:.2f}")
            
            # Track best model
            if test_elbo > prior_results["best_elbo"]:
                prior_results["best_elbo"] = test_elbo
                prior_results["best_model_path"] = f"{run_dir}/vae_{prior_type}_final.pt"
        
        # Compute statistics
        import numpy as np
        prior_results["mean_elbo"] = np.mean(prior_results["elbos"])
        prior_results["std_elbo"] = np.std(prior_results["elbos"])
        
        print(f"\n{prior_type.upper()} Results:")
        print(f"Test ELBO: {prior_results['mean_elbo']:.2f} ± {prior_results['std_elbo']:.2f}")
        
        # Generate plots for best model
        print(f"\nGenerating plots for {prior_type} prior...")
        best_model, _ = load_model(prior_results["best_model_path"], device)
        
        create_all_plots_for_prior(
            model=best_model,
            data_loader=train_loader,
            device=device,
            prior_name=prior_type,
            history=prior_results["histories"][0],  # Use first run history
            output_dir=str(figures_dir),
        )
        
        all_results[prior_type] = {
            "mean": prior_results["mean_elbo"],
            "std": prior_results["std_elbo"],
            "all_runs": prior_results["elbos"],
        }
    
    # Generate comparison plot
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    
    plot_elbo_comparison(all_results, save_path=f"{figures_dir}/elbo_comparison.png")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Prior':<20} {'Mean ELBO':<15} {'Std':<10} {'Runs'}")
    print("-" * 55)
    for prior, res in all_results.items():
        runs_str = ", ".join([f"{e:.1f}" for e in res["all_runs"]])
        print(f"{prior.upper():<20} {res['mean']:<15.2f} {res['std']:<10.2f} [{runs_str}]")
    print("=" * 70)
    
    # Save results to JSON
    results_path = Path(output_dir) / "part_a_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_runs": n_runs,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Part A experiments: VAE with different priors"
    )
    parser.add_argument(
        "--prior", type=str, default=None,
        choices=["gaussian", "mog", "flow"],
        help="Run experiments for specific prior only"
    )
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs per prior for mean/std")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per run")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./reports")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (1 run, 5 epochs)")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.n_runs = 1
        args.epochs = 5
        print("*** QUICK TEST MODE: 1 run, 5 epochs ***")
    
    # Determine priors to run
    priors = [args.prior] if args.prior else None
    
    run_part_a_experiments(
        n_runs=args.n_runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        priors=priors,
    )


if __name__ == "__main__":
    main()
