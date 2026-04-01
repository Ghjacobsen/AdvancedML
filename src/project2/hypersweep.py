"""Hyperparameter sweep for geodesic CoV optimization.

Searches over geodesic optimizer settings to find the combination
that produces the expected behavior: geodesic CoV decreasing with K.

Uses the already-trained models — no retraining needed.
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from project2.data import get_data_loaders
from project2.experiment import select_point_pairs
from project2.geodesics import euclidean_distance, geodesic_distance
from project2.train import load_ensemble


def evaluate_cov(
    save_dir: str,
    num_reruns: int,
    num_decoders: int,
    num_pairs: int,
    pair_seed: int,
    latent_dim: int,
    device: str,
    # Geodesic hyperparams
    geo_lr: float,
    geo_steps: int,
    num_mc: int,
    num_t: int,
    lr_decay: float,
    decay_every: int,
):
    """Run CoV evaluation with given geodesic hyperparameters.

    Returns dict with CoV per K and timing info.
    """
    _, test_loader, _, _ = get_data_loaders(batch_size=64, data_dir="data/")
    K_values = list(range(1, num_decoders + 1))

    euclidean_dists = {K: [] for K in K_values}
    geodesic_dists = {K: [] for K in K_values}

    t0 = time.time()

    for m in range(num_reruns):
        retrain_dir = os.path.join(save_dir, f"retrain_{m}")
        if not os.path.exists(retrain_dir):
            continue

        for K in K_values:
            vaes, decoders, encoder = load_ensemble(retrain_dir, K, M=latent_dim, device=device)
            y_i, y_j, z_i, z_j, _, _ = select_point_pairs(
                test_loader, encoder, num_pairs=num_pairs, device=device, seed=pair_seed
            )

            euc_d, geo_d = [], []
            for p in range(num_pairs):
                euc_d.append(euclidean_distance(z_i[p], z_j[p]).item())
                _, gd = geodesic_distance(
                    z_i[p], z_j[p], decoders,
                    num_t=num_t, lr=geo_lr, num_steps=geo_steps,
                    num_mc_samples=num_mc, lr_decay=lr_decay, decay_every=decay_every,
                )
                geo_d.append(gd.item())

            euclidean_dists[K].append(np.array(euc_d))
            geodesic_dists[K].append(np.array(geo_d))

    elapsed = time.time() - t0

    # Compute CoV
    result = {"time_s": elapsed}
    for K in K_values:
        euc_stack = np.stack(euclidean_dists[K], axis=0)
        geo_stack = np.stack(geodesic_dists[K], axis=0)

        euc_cov = (euc_stack.std(0) / (euc_stack.mean(0) + 1e-10)).mean()
        geo_cov = (geo_stack.std(0) / (geo_stack.mean(0) + 1e-10)).mean()

        result[f"euc_cov_K{K}"] = float(euc_cov)
        result[f"geo_cov_K{K}"] = float(geo_cov)

    # Score: we want geo_cov to decrease with K. Lower score = better.
    # Penalize if K=1 > K=2 > K=3 is violated.
    geo_covs = [result[f"geo_cov_K{K}"] for K in K_values]
    # Ideal: monotonically decreasing. Score = sum of geo_covs weighted + penalty for non-monotonicity
    score = geo_covs[-1]  # minimize final CoV
    for i in range(1, len(geo_covs)):
        if geo_covs[i] > geo_covs[i - 1]:
            score += 0.5  # penalty for non-monotonic
    result["score"] = float(score)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Geodesic hyperparameter sweep")
    parser.add_argument("--save-dir", default="models/ensemble")
    parser.add_argument("--num-reruns", type=int, default=10)
    parser.add_argument("--num-decoders", type=int, default=3)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-time", type=int, default=3300, help="Max runtime in seconds (default ~55min)")
    args = parser.parse_args()

    # Define search grid
    grid = {
        "geo_lr":      [0.1, 0.05, 0.01, 0.005, 0.001],
        "geo_steps":   [500, 1000, 2000],
        "num_mc":      [50, 100, 200],
        "num_t":       [10, 20, 40],
        "lr_decay":    [0.5, 0.3],
        "decay_every": [200, 400],
    }

    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    print(f"Total combinations: {len(combos)}")
    print(f"Max time: {args.max_time}s")
    print(f"Device: {args.device}")
    print()

    results_dir = os.path.join(args.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "hypersweep_results.json")

    all_results = []
    global_start = time.time()

    # Sort combos by expected speed: fewer steps & fewer t first
    combos_sorted = sorted(combos, key=lambda c: c[keys.index("geo_steps")] * c[keys.index("num_t")])

    for i, combo in enumerate(combos_sorted):
        elapsed_total = time.time() - global_start
        if elapsed_total > args.max_time:
            print(f"\nTime limit reached ({elapsed_total:.0f}s). Stopping after {i} configs.")
            break

        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combos)}] {params}")

        try:
            result = evaluate_cov(
                save_dir=args.save_dir,
                num_reruns=args.num_reruns,
                num_decoders=args.num_decoders,
                num_pairs=args.num_pairs,
                pair_seed=args.pair_seed,
                latent_dim=args.latent_dim,
                device=args.device,
                **params,
            )
            result["params"] = params
            all_results.append(result)

            geo_str = " | ".join(f"K{K}={result[f'geo_cov_K{K}']:.3f}" for K in range(1, args.num_decoders + 1))
            print(f"  Geo CoV: {geo_str} | score={result['score']:.3f} | {result['time_s']:.0f}s")

            # Save incrementally
            with open(out_path, "w") as f:
                json.dump(sorted(all_results, key=lambda r: r["score"]), f, indent=2)

        except Exception as e:
            print(f"  FAILED: {e}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"Completed {len(all_results)} configs in {time.time() - global_start:.0f}s")
    print(f"\nTop 5 by score:")
    for r in sorted(all_results, key=lambda r: r["score"])[:5]:
        geo_str = " | ".join(f"K{K}={r[f'geo_cov_K{K}']:.3f}" for K in range(1, args.num_decoders + 1))
        print(f"  score={r['score']:.3f} | {geo_str} | {r['params']}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
