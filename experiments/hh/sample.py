"""Simulate the Hodgkin-Huxley training or test set (Hydra entry point).

Draws parameters from the mixed prior, simulates them with jaxley, and keeps the
summary statistics of the simulations that produced no NaN. Writes
``artifacts/hh/<data_config.output_file>``. The defaults in
``configs/data_config/hh_model.yaml`` write the 10k test set; the training set is
the same call at the training seed:

    uv run python experiments/hh/sample.py
    uv run python experiments/hh/sample.py data_config.n_samples=100000 \
        data_config.seed=42 data_config.output_file=data/jaxley_data_100k.pkl
"""

import pickle

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch
from model import HH_n_continous, HH_prior, HH_sim, HH_summary_stats, HH_transform
from omegaconf import DictConfig

from mnpe_paper.paths import model_artifacts_dir

ARTIFACTS_DIR = model_artifacts_dir("hh")


def sample_and_simulate(n_needed, batch_size, save_raw_data=False):
    """Sample parameters from prior, simulate, compute summary stats.
    Returns only valid (non-NaN) samples. May return fewer than n_needed."""
    thetas_raw = HH_prior.sample(sample_shape=(n_needed,))
    thetas = jnp.concatenate(
        [
            HH_transform.inverse(jnp.array(thetas_raw[:, :HH_n_continous])),
            jnp.array(thetas_raw[:, HH_n_continous:]),
        ],
        axis=1,
    )

    valid_thetas = []
    valid_ss = []
    valid_raw = []

    for i in range(0, n_needed, batch_size):
        batch = thetas[i : i + batch_size]
        sim_batch = HH_sim(batch)
        ss_batch = jax.vmap(HH_summary_stats)(sim_batch)

        ss_np = np.array(ss_batch)
        valid_mask = ~np.isnan(ss_np).any(axis=1)

        valid_thetas.append(np.array(batch)[valid_mask])
        valid_ss.append(ss_np[valid_mask])
        if save_raw_data:
            valid_raw.append(np.array(sim_batch)[valid_mask])

        del sim_batch, ss_batch
        jax.clear_caches()

    results = (
        np.concatenate(valid_thetas),
        np.concatenate(valid_ss),
    )
    if save_raw_data:
        results = results + (np.concatenate(valid_raw),)
    else:
        results = results + (None,)
    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.data_config.seed
    n_samples = cfg.data_config.n_samples
    batch_size = cfg.data_config.batch_size
    save_raw_data = cfg.data_config.save_raw_data

    torch.manual_seed(seed)

    all_thetas = []
    all_ss = []
    all_raw = []
    n_collected = 0
    n_total_simulated = 0
    round_idx = 0

    while n_collected < n_samples:
        n_needed = n_samples - n_collected
        # sample extra to account for expected NaN rate
        if round_idx > 0 and n_total_simulated > 0:
            nan_rate = 1 - n_collected / n_total_simulated
            n_to_sample = int(n_needed / max(1 - nan_rate, 0.1))
        else:
            n_to_sample = n_needed

        print(f"Round {round_idx}: sampling {n_to_sample} (need {n_needed} more valid samples)")
        thetas_valid, ss_valid, raw_valid = sample_and_simulate(
            n_to_sample, batch_size, save_raw_data=save_raw_data
        )
        n_total_simulated += n_to_sample

        all_thetas.append(thetas_valid)
        all_ss.append(ss_valid)
        if save_raw_data:
            all_raw.append(raw_valid)

        n_collected += len(thetas_valid)
        n_filtered = n_to_sample - len(thetas_valid)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} NaN samples, collected {n_collected}/{n_samples}")
        round_idx += 1

    thetas_out = np.concatenate(all_thetas)[:n_samples]
    ss_out = np.concatenate(all_ss)[:n_samples]

    data_dict = {
        "thetas": thetas_out,
        "x": ss_out,
    }
    if save_raw_data:
        raw_out = np.concatenate(all_raw)[:n_samples]
        data_dict["raw_data"] = raw_out

    output_path = ARTIFACTS_DIR / cfg.data_config.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Saved {n_samples} samples to {output_path} "
          f"(simulated {n_total_simulated} total, {round_idx} round(s))")


if __name__ == "__main__":
    main()
