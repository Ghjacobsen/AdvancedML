"""Full experiment orchestration for ensemble VAE geometry.

Three main entry points:
  1. train   — Train all M retrainings × K decoders (or a single retrain)
  2. geodesics — Compute geodesics for Part A and Part B
  3. cov     — Compute CoV analysis across retrainings

Designed so that each stage can be run independently, loading checkpoints
from disk.  After training, you can re-run geodesics/cov/plots with
different hyperparameters without retraining.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from project2.data import get_data_loaders
from project2.geodesics import compute_geodesic, euclidean_distance, geodesic_distance, ensemble_curve_energy, curve_energy
from project2.train import load_ensemble, train_ensemble
from project2.visualize import plot_geodesics, plot_cov


def select_point_pairs(
    test_loader: torch.utils.data.DataLoader,
    encoder,
    num_pairs: int,
    device: str,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select fixed test point pairs for geodesic/CoV evaluation.

    Encodes the full test set, selects random pairs of latent means.
    Returns data-space points (for reference) and latent means.

    Returns:
        (y_i, y_j, z_i, z_j): Test images and their latent means,
            each of shape (num_pairs, ...).
    """
    all_x, all_labels = [], []
    with torch.no_grad():
        for x, lab in test_loader:
            all_x.append(x)
            all_labels.append(lab)
    all_x = torch.cat(all_x, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0)

    # Encode to get latent means
    with torch.no_grad():
        q = encoder(all_x)
        all_z = q.mean  # (N, M)

    # Select random pairs
    rng = np.random.RandomState(seed)
    n = all_x.shape[0]
    idx_i = rng.choice(n, size=num_pairs, replace=False)
    idx_j = rng.choice(n, size=num_pairs, replace=False)
    # Make sure i != j
    for p in range(num_pairs):
        while idx_j[p] == idx_i[p]:
            idx_j[p] = rng.choice(n)

    y_i = all_x[idx_i]
    y_j = all_x[idx_j]
    z_i = all_z[idx_i]
    z_j = all_z[idx_j]
    labels_i = all_labels[idx_i]
    labels_j = all_labels[idx_j]

    return y_i, y_j, z_i, z_j, all_z, all_labels


def run_training(args):
    """Train all M retrainings × K decoders."""
    device = args.device
    train_loader, _, _, _ = get_data_loaders(batch_size=args.batch_size, data_dir=args.data_dir)

    for m in range(args.num_reruns):
        print(f"\n{'#'*60}")
        print(f"  RETRAIN {m+1}/{args.num_reruns}")
        print(f"{'#'*60}")

        # Set seed per retrain for reproducibility
        torch.manual_seed(m * 1000)

        save_dir = os.path.join(args.save_dir, f"retrain_{m}")
        train_ensemble(
            num_decoders=args.num_decoders,
            data_loader=train_loader,
            M=args.latent_dim,
            epochs_per_decoder=args.epochs,
            lr=args.lr,
            device=device,
            save_dir=save_dir,
            verbose=True,
        )

    print(f"\nAll training complete. Models saved to {args.save_dir}/")


def run_geodesics(args):
    """Compute geodesics for Part A (single decoder) and Part B (ensemble)."""
    device = args.device
    _, test_loader, _, _ = get_data_loaders(batch_size=args.batch_size, data_dir=args.data_dir)

    results_dir = os.path.join(args.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Use retrain 0 for Part A / geodesic plots
    retrain_dir = os.path.join(args.save_dir, f"retrain_{args.retrain_idx}")

    for K in range(1, args.num_decoders + 1):
        print(f"\n{'='*60}")
        print(f"  Computing geodesics with K={K} decoders (retrain {args.retrain_idx})")
        print(f"{'='*60}")

        vaes, decoders, encoder = load_ensemble(retrain_dir, K, M=args.latent_dim, device=device)

        # Select point pairs using encoder from this retrain
        y_i, y_j, z_i, z_j, all_z, all_labels = select_point_pairs(
            test_loader, encoder, num_pairs=args.num_curves, device=device, seed=args.pair_seed
        )

        curves = []
        for p in range(args.num_curves):
            print(f"  Pair {p+1}/{args.num_curves}")
            curve = compute_geodesic(
                z_i[p],
                z_j[p],
                decoders,
                num_t=args.num_t,
                lr=args.geo_lr,
                num_steps=args.geo_steps,
                num_mc_samples=args.num_mc,
                verbose=args.verbose,
            )
            curves.append(curve.detach().cpu())

        curves = torch.stack(curves, dim=0)  # (num_curves, num_t+2, M)

        # Save results
        torch.save(
            {
                "curves": curves,
                "z_i": z_i.cpu(),
                "z_j": z_j.cpu(),
                "all_z": all_z.cpu(),
                "all_labels": all_labels.cpu(),
                "K": K,
                "retrain_idx": args.retrain_idx,
            },
            os.path.join(results_dir, f"geodesics_K{K}_retrain{args.retrain_idx}.pt"),
        )

        # Plot
        plot_geodesics(
            all_z.cpu().numpy(),
            all_labels.cpu().numpy(),
            curves.numpy(),
            title=f"Geodesics (K={K} decoder{'s' if K > 1 else ''})",
            save_path=os.path.join(results_dir, f"geodesics_K{K}.pdf"),
        )

    print(f"\nGeodesic results saved to {results_dir}/")


def run_cov(args):
    """Compute CoV analysis across retrainings for K=1,2,3."""
    device = args.device
    _, test_loader, _, _ = get_data_loaders(batch_size=args.batch_size, data_dir=args.data_dir)

    results_dir = os.path.join(args.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    num_pairs = args.num_cov_pairs
    K_values = list(range(1, args.num_decoders + 1))

    # Storage: distances[K][m] = list of distances for each pair
    euclidean_dists = {K: [] for K in K_values}  # [K] -> list of (num_pairs,) arrays
    geodesic_dists = {K: [] for K in K_values}

    for m in range(args.num_reruns):
        print(f"\n{'#'*60}")
        print(f"  RETRAIN {m}/{args.num_reruns} — CoV evaluation")
        print(f"{'#'*60}")

        retrain_dir = os.path.join(args.save_dir, f"retrain_{m}")
        if not os.path.exists(retrain_dir):
            print(f"  Skipping retrain {m} — directory not found")
            continue

        for K in K_values:
            print(f"\n  K={K} decoders")

            vaes, decoders, encoder = load_ensemble(retrain_dir, K, M=args.latent_dim, device=device)

            # Encode test points — use THIS retrain's encoder
            y_i, y_j, z_i, z_j, _, _ = select_point_pairs(
                test_loader, encoder, num_pairs=num_pairs, device=device, seed=args.pair_seed
            )

            euc_d = []
            geo_d = []
            for p in range(num_pairs):
                # Euclidean
                ed = euclidean_distance(z_i[p], z_j[p]).item()
                euc_d.append(ed)

                # Geodesic
                _, gd = geodesic_distance(
                    z_i[p],
                    z_j[p],
                    decoders,
                    num_t=args.num_t,
                    lr=args.geo_lr,
                    num_steps=args.geo_steps,
                    num_mc_samples=args.num_mc,
                )
                geo_d.append(gd.item())
                print(f"    Pair {p+1}/{num_pairs}: euc={ed:.4f}, geo={gd.item():.4f}")

            euclidean_dists[K].append(np.array(euc_d))
            geodesic_dists[K].append(np.array(geo_d))

    # Compute CoV for each K
    # dists[K] is list of M arrays, each (num_pairs,)
    # Stack to (M, num_pairs), compute CoV across axis=0

    cov_results = {}
    for K in K_values:
        euc_stack = np.stack(euclidean_dists[K], axis=0)  # (M, num_pairs)
        geo_stack = np.stack(geodesic_dists[K], axis=0)

        euc_cov_per_pair = euc_stack.std(axis=0) / (euc_stack.mean(axis=0) + 1e-10)
        geo_cov_per_pair = geo_stack.std(axis=0) / (geo_stack.mean(axis=0) + 1e-10)

        cov_results[K] = {
            "euclidean_cov_mean": float(euc_cov_per_pair.mean()),
            "euclidean_cov_std": float(euc_cov_per_pair.std()),
            "geodesic_cov_mean": float(geo_cov_per_pair.mean()),
            "geodesic_cov_std": float(geo_cov_per_pair.std()),
            "euclidean_cov_per_pair": euc_cov_per_pair.tolist(),
            "geodesic_cov_per_pair": geo_cov_per_pair.tolist(),
        }

        print(f"\nK={K}: Euclidean CoV = {euc_cov_per_pair.mean():.4f} ± {euc_cov_per_pair.std():.4f}")
        print(f"K={K}: Geodesic  CoV = {geo_cov_per_pair.mean():.4f} ± {geo_cov_per_pair.std():.4f}")

    # Save raw distances for later re-plotting
    torch.save(
        {"euclidean": euclidean_dists, "geodesic": geodesic_dists, "cov": cov_results},
        os.path.join(results_dir, "cov_distances.pt"),
    )

    # Save summary
    with open(os.path.join(results_dir, "cov_results.json"), "w") as f:
        json.dump(cov_results, f, indent=2)

    # Plot
    plot_cov(cov_results, save_path=os.path.join(results_dir, "cov_plot.pdf"))

    print(f"\nCoV results saved to {results_dir}/")


def run_plot(args):
    """Re-generate plots from saved results (no retraining/recomputing)."""
    results_dir = os.path.join(args.save_dir, "results")

    # Re-plot geodesics
    K_values = list(range(1, args.num_decoders + 1))
    for K in K_values:
        path = os.path.join(results_dir, f"geodesics_K{K}_retrain{args.retrain_idx}.pt")
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            plot_geodesics(
                data["all_z"].numpy(),
                data["all_labels"].numpy(),
                data["curves"].numpy(),
                title=f"Geodesics (K={K} decoder{'s' if K > 1 else ''})",
                save_path=os.path.join(results_dir, f"geodesics_K{K}.pdf"),
            )
            print(f"Re-plotted geodesics K={K}")

    # Re-plot CoV
    cov_path = os.path.join(results_dir, "cov_results.json")
    if os.path.exists(cov_path):
        with open(cov_path) as f:
            cov_results_raw = json.load(f)
        # Convert string keys back to int
        cov_results = {int(k): v for k, v in cov_results_raw.items()}
        plot_cov(cov_results, save_path=os.path.join(results_dir, "cov_plot.pdf"))
        print("Re-plotted CoV")


def main():
    parser = argparse.ArgumentParser(description="Ensemble VAE Geometry Experiments")
    parser.add_argument(
        "mode",
        choices=["train", "geodesics", "cov", "plot", "all"],
        help="Which stage to run.",
    )

    # Paths
    parser.add_argument("--save-dir", default="models/ensemble", help="Base directory for checkpoints & results")
    parser.add_argument("--data-dir", default="data/", help="MNIST data directory")

    # Model
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-decoders", type=int, default=3, help="Max K (ensemble size)")
    parser.add_argument("--num-reruns", type=int, default=10, help="M — number of retrainings for CoV")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Geodesics
    parser.add_argument("--num-curves", type=int, default=25, help="Number of geodesic pairs to plot")
    parser.add_argument("--num-cov-pairs", type=int, default=10, help="Number of pairs for CoV analysis")
    parser.add_argument("--num-t", type=int, default=20, help="Interior curve points")
    parser.add_argument("--geo-lr", type=float, default=0.01, help="Geodesic optimizer LR")
    parser.add_argument("--geo-steps", type=int, default=1000, help="Geodesic optimizer steps")
    parser.add_argument("--num-mc", type=int, default=50, help="MC samples for ensemble energy")
    parser.add_argument("--pair-seed", type=int, default=42, help="Seed for test pair selection")
    parser.add_argument("--retrain-idx", type=int, default=0, help="Which retrain to use for geodesic plots")

    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("# Ensemble VAE Geometry Experiment")
    print(f"# Mode: {args.mode}")
    for k, v in sorted(vars(args).items()):
        print(f"#   {k} = {v}")
    print()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "geodesics":
        run_geodesics(args)
    elif args.mode == "cov":
        run_cov(args)
    elif args.mode == "plot":
        run_plot(args)
    elif args.mode == "all":
        run_training(args)
        run_geodesics(args)
        run_cov(args)


if __name__ == "__main__":
    main()
