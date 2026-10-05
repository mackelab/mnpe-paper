"""Train MNPE on the Hodgkin-Huxley model, then evaluate it (Hydra entry point).

Writes ``artifacts/hh/{n_train}/run_{seed}/``: the trained inference object and
``mse_results_{dropout_rate}.pkl`` (posterior-predictive MSE, SBC ranks of the
marginals, discrete calibration). Training and test data are read from
``artifacts/hh/data/``, written there by ``sample.py``. Needs jaxley and Hydra; see
this directory's README.

    uv run python experiments/hh/inference.py inference_config.n_train_samples=100000 \
        inference_config.seed=2
"""

import os
import pickle

import hydra
import numpy as np
import torch
from evaluate import evaluate_posterior
from model import HH_n_continous
from omegaconf import DictConfig
from sbi.diagnostics import run_sbc
from sbi.inference import MNPE
from sbi.neural_nets import posterior_nn

from mnpe_paper.calibration import eod_from_ranks, run_discrete_calibration
from mnpe_paper.paths import model_artifacts_dir

ARTIFACTS_DIR = model_artifacts_dir("hh")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # set seed (generate random one if not specified)
    seed = cfg.inference_config.seed
    if seed is None:
        seed = int(torch.randint(0, 2**32, (1,)).item())
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_train = cfg.inference_config.n_train_samples

    dropout_rate = cfg.inference_config.dropout_rate

    # output directory: artifacts/hh/{n_train}/run_{seed}/
    output_dir = os.path.join(ARTIFACTS_DIR, str(n_train), f"run_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    output_file_stem, output_file_ext = os.path.splitext(cfg.inference_config.output_file)
    output_path = os.path.join(output_dir, f"{output_file_stem}_{dropout_rate}{output_file_ext}")

    if os.path.exists(output_path):
        print(f"Posterior exists at {output_path}, skipping training.")
        with open(output_path, "rb") as f:
            inference = pickle.load(f)
        posterior = inference.build_posterior()
    else:
        data_path = os.path.join(ARTIFACTS_DIR, cfg.inference_config.data_file)
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)

        x = torch.tensor(np.array(data_dict["x"])[:n_train], dtype=torch.float32)
        thetas = torch.tensor(np.array(data_dict["thetas"])[:n_train], dtype=torch.float32)

        density_estimator = posterior_nn(
            model="mnpe",
            hidden_features=cfg.inference_config.hidden_features,
            num_transforms=cfg.inference_config.num_transforms,
            num_bins=cfg.inference_config.num_bins,
            hidden_layers=cfg.inference_config.hidden_layers,
            num_blocks=cfg.inference_config.num_blocks,
            z_score_theta='independent',
            z_score_x='independent',
            dropout_probability=cfg.inference_config.dropout_rate,
        )

        inference = MNPE(density_estimator=density_estimator)

        _ = inference.append_simulations(thetas, x).train(
            max_num_epochs=cfg.inference_config.max_num_epochs,
            training_batch_size=cfg.inference_config.training_batch_size,
        )

        print("finished training.")
        posterior = inference.build_posterior()

        with open(output_path, "wb") as f:
            pickle.dump(inference, f)

    test_data_path = os.path.join(ARTIFACTS_DIR, cfg.inference_config.test_data_file)
    eval_save_path = os.path.join(output_dir, f"mse_results_{dropout_rate}.pkl")
    mse_results = evaluate_posterior(
        test_data_path=test_data_path,
        posterior=posterior,
        n_samples=cfg.inference_config.eval_n_samples,
        n_test=cfg.inference_config.eval_n_test,
    )
    mse_results["n_train_samples"] = n_train
    mse_results["seed"] = seed

    with open(test_data_path, "rb") as f:
        data_dict = pickle.load(f)

    x_all = torch.tensor(np.array(data_dict["x"]), dtype=torch.float32)
    thetas_all = torch.tensor(np.array(data_dict["thetas"]), dtype=torch.float32)
    num_posterior_samples = cfg.inference_config.num_posterior_samples
    n_sbc = cfg.inference_config.num_samples_sbc
    n_sbc_discrete = cfg.inference_config.num_samples_sbc_discrete

    print("running SBC (marginals)...")
    ranks_marginals, _ = run_sbc(
        thetas_all[:n_sbc], x_all[:n_sbc], posterior,
        num_posterior_samples=num_posterior_samples,
        reduce_fns="marginals",
        show_progress_bar=False,
        use_batched_sampling=True,
    )
    sbc_ece_marginals = eod_from_ranks(ranks_marginals, num_posterior_samples)


    # Discrete calibration on the per-variable marginals. The stored results in
    # results/hh/ score the autoregressive factors instead; see the README.
    print("running discrete calibration...")
    discrete_indices = list(range(HH_n_continous, thetas_all.shape[1]))
    ece_discrete, probs_discrete, true_vals_discrete = run_discrete_calibration(
        thetas_all[:n_sbc_discrete], x_all[:n_sbc_discrete], posterior,
        discrete_indices=discrete_indices,
        num_bins=cfg.inference_config.num_discrete_sbc_bins,
        probability_type="marginal",
    )

    mse_results["num_posterior_samples"] = num_posterior_samples
    mse_results["sbc_ece_marginals"] = sbc_ece_marginals
    mse_results["ranks_marginals"] = ranks_marginals
    mse_results["ece_discrete"] = ece_discrete
    mse_results["probs_discrete"] = probs_discrete
    mse_results["true_vals_discrete"] = true_vals_discrete
    mse_results["discrete_indices"] = discrete_indices
    mse_results["num_discrete_bins"] = cfg.inference_config.num_discrete_sbc_bins

    with open(eval_save_path, "wb") as f:
        pickle.dump(mse_results, f)
    print(f"Saved evaluation results to {eval_save_path}")


if __name__ == "__main__":
    main()
