"""Posterior-predictive MSE on summary statistics, called by ``inference.py``."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import torch
from model import HH_sim, HH_summary_stats


def evaluate_posterior(test_data_path, posterior, n_samples=10, n_test=None, verbose=True):
    """Evaluate posterior by computing MSE on summary stats between
    posterior predictive simulations and test data.

    Args:
        test_data_path: Path to test data pickle file.
        posterior: SBI posterior object.
        n_samples: Number of posterior samples per test datapoint.
        n_test: Number of test datapoints to use (None = all).
    """
    with open(test_data_path, "rb") as f:
        data_dict = pickle.load(f)

    x = torch.tensor(np.array(data_dict["x"]), dtype=torch.float32)

    if n_test is None:
        n_test = x.shape[0]

    # Collect all posterior samples sequentially (context changes per point)
    all_samples = []
    for i in range(n_test):
        samples = posterior.sample((n_samples,), x=x[i], show_progress_bars=False)
        all_samples.append(samples)
        if verbose and i % 100 == 0:
            print(f"Sampled {i} / {n_test}")

    # One batched simulation call instead of n_test separate calls
    all_samples_flat = jnp.array(torch.cat(all_samples, dim=0))  # (n_test*n_samples, n_params)
    x_post_flat = HH_sim(all_samples_flat)
    ss_flat = jax.vmap(HH_summary_stats)(x_post_flat)
    summary_stats = np.array(ss_flat).reshape(n_test, n_samples, -1)

    # MSE per sample per datapoint (NaN-robust)
    squared_errors = (summary_stats - x.numpy()[:n_test, None, :]) ** 2
    mse_per_sample = np.nanmean(squared_errors, axis=2)  # (n_datapoints, n_samples)
    mse_per_point = np.nanmean(mse_per_sample, axis=1)  # (n_datapoints,)

    mse_mean = np.nanmean(mse_per_point)
    mse_std = np.nanstd(mse_per_point)
    mse_p5 = np.nanpercentile(mse_per_point, 5)
    mse_p50 = np.nanpercentile(mse_per_point, 50)
    mse_p95 = np.nanpercentile(mse_per_point, 95)
    n_nan_points = np.isnan(mse_per_point).sum()

    if verbose:
        print(f"Mean MSE: {mse_mean:.4f} (std: {mse_std:.4f})")
        print(f"Percentiles — 5%: {mse_p5:.4f}, 50%: {mse_p50:.4f}, 95%: {mse_p95:.4f}")
        print(f"Excluded {n_nan_points} fully-NaN datapoints")

    mse_results = {
        "mse_per_sample": mse_per_sample,
        "mse_per_point": mse_per_point,
        "mean": mse_mean,
        "std": mse_std,
        "p5": mse_p5,
        "p50": mse_p50,
        "p95": mse_p95,
    }

    return mse_results
