"""Runtime comparison: amortized MNPE vs. exact-marginalized PyMC.

Backs Section 4.2 and Table D-1 of Appendix D.2. Amortization is a central claim
of the method, so the comparison is stated as a cost model with an explicit
break-even rather than as a single wall-clock number.

Cost model:
  MNPE one-off   = simulation of the training set + training
  MNPE per-obs   = posterior sampling (10k samples) at a new observation
  MCMC per-obs   = one SMC run on the exactly marginalized 3D target
  break-even n*  = one_off / (mcmc_per_obs - mnpe_per_obs)

MCMC is SMC, not NUTS. NUTS does not converge on this target (R-hat ~1.6, bulk
ESS ~6), so timing it would report the cost of running a sampler rather than the
cost of obtaining a posterior. See ../README.md, "Why the reference is built
this way".

Two settings, both honest:
  single   — one SMC run, the minimum that yields a posterior
  checked  — five independent runs, which is what the reference artifact uses
             and what a careful practitioner needs for a convergence diagnostic

Stages (composable; results accumulate in runtime_comparison_results.json):
  uv run python runtime_comparison.py --mnpe
  uv run python runtime_comparison.py --mcmc single
  uv run python runtime_comparison.py --mcmc checked
  uv run python runtime_comparison.py --report   # table + break-even -> CSV/MD
"""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import torch

from mnpe_paper.paths import model_artifacts_dir, model_results_dir

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
RESULTS_FILE = ARTIFACTS_DIR / "runtime_comparison_results.json"

BUDGETS = [10_000, 100_000]
N_REPS = 3
N_POSTERIOR_SAMPLES = 10_000

MCMC_SETTINGS = {
    "single": {"draws": 6000, "chains": 4, "runs": 1},
    "checked": {"draws": 6000, "chains": 4, "runs": 5},
}


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def get_hardware():
    def sysctl(key):
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        return out.stdout.strip()

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "cores": sysctl("hw.ncpu"),
        "platform": sys.platform,
    }


# ══════════════════════════════════════════════════════════════════════
# MNPE: simulation + training + sampling
# ══════════════════════════════════════════════════════════════════════


def time_mnpe():
    from sbi.inference import MNPE
    from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters

    from mnpe_paper.models.queue.config import (
        build_sbi_prior,
        generate_reference_observation,
        generate_training_data,
        transform_x,
    )
    from mnpe_paper.models.queue.eval import build_tuned_mnpe
    from mnpe_paper.models.queue.spec import TUNED_HPARAMS as tuned

    prior = build_sbi_prior()
    obs = generate_reference_observation(seed=42)
    x_o = transform_x(torch.tensor(obs["x_array"], dtype=torch.float32).unsqueeze(0))

    out = {"budgets": {}, "tuned_hparams": tuned, "n_posterior_samples": N_POSTERIOR_SAMPLES}

    for budget in BUDGETS:
        print(f"\n── MNPE budget {budget:,} ──")

        t0 = time.perf_counter()
        theta_train, x_train_raw = generate_training_data(budget, seed=42)
        sim_time = time.perf_counter() - t0
        x_train = transform_x(x_train_raw)
        print(f"  simulation ({budget:,} draws): {sim_time:.1f}s")

        train_times, sample_times = [], []
        for rep in range(N_REPS):
            torch.manual_seed(rep * 1000)
            trainer = MNPE(prior=prior, density_estimator=build_tuned_mnpe())
            trainer.append_simulations(theta_train, x_train)

            t0 = time.perf_counter()
            trainer.train(
                training_batch_size=tuned["training_batch_size"],
                learning_rate=tuned["learning_rate"],
            )
            train_times.append(time.perf_counter() - t0)

            post = trainer.build_posterior(
                posterior_parameters=DirectPosteriorParameters(enable_transform=False)
            )
            t0 = time.perf_counter()
            post.sample((N_POSTERIOR_SAMPLES,), x=x_o, show_progress_bars=False)
            sample_times.append(time.perf_counter() - t0)
            print(
                f"  rep {rep}: train {train_times[-1]:.1f}s, "
                f"sample {N_POSTERIOR_SAMPLES:,} {sample_times[-1]:.2f}s"
            )

        out["budgets"][str(budget)] = {
            "sim_time_s": sim_time,
            "train_times_s": train_times,
            "sample_times_s": sample_times,
        }

    results = load_results()
    results["mnpe"] = out
    results["hardware"] = get_hardware()
    save_results(results)
    print(f"\nSaved MNPE timings to {RESULTS_FILE.name}")


# ══════════════════════════════════════════════════════════════════════
# MCMC: SMC on the exactly marginalized PyMC model, in process
# ══════════════════════════════════════════════════════════════════════


def time_mcmc(setting):
    """Time SMC on the exactly marginalized target, in process."""
    import warnings

    import arviz as az
    import pymc as pm

    from mnpe_paper.models.queue.config import (
        build_pymc_marginalized_model,
        generate_reference_observation,
    )

    cfg = MCMC_SETTINGS[setting]
    observations = generate_reference_observation(seed=42)
    model = build_pymc_marginalized_model(observations)

    print(
        f"── MCMC/SMC ({setting}): {cfg['runs']} run(s) x {cfg['chains']} "
        f"chains x {cfg['draws']:,} draws, 25 configs marginalized exactly ──"
    )

    # Build cost is paid once per observation in a real workflow, so it is
    # timed and reported separately rather than hidden or double counted.
    t0 = time.perf_counter()
    with model:
        pm.sample_smc(draws=200, chains=2, random_seed=0, progressbar=False)
    compile_time = time.perf_counter() - t0
    print(f"  compile + warm-up (once per process): {compile_time:.1f}s")

    run_times, rhats = [], []
    for run in range(cfg["runs"]):
        t0 = time.perf_counter()
        with model, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample_smc(
                draws=cfg["draws"],
                chains=cfg["chains"],
                random_seed=42 + run,
                progressbar=False,
            )
        run_times.append(time.perf_counter() - t0)
        summary = az.summary(idata, var_names=["gamma", "mu1", "mu2"])
        rhats.append(float(summary["r_hat"].max()))
        print(f"  run {run}: {run_times[-1]:.1f}s, max R-hat {rhats[-1]:.4f}")

    total = float(np.sum(run_times))
    results = load_results()
    results.setdefault("mcmc", {})[setting] = {
        **cfg,
        "wall_time_s": total,
        "run_times_s": run_times,
        "compile_time_s": compile_time,
        "max_rhat": max(rhats),
        "n_configs_marginalized": 25,
        "converged": bool(max(rhats) <= 1.01),
        "sampler": "SMC",
    }
    save_results(results)
    print(
        f"  total {total:.1f}s for {cfg['runs']} run(s); "
        f"max R-hat {max(rhats):.4f}"
    )
    print(f"Saved MCMC ({setting}) timings to {RESULTS_FILE.name}")


# ══════════════════════════════════════════════════════════════════════
# Report: table + break-even
# ══════════════════════════════════════════════════════════════════════


def report():
    import pandas as pd

    results = load_results()
    mnpe, mcmc = results.get("mnpe"), results.get("mcmc", {})
    if not mnpe or not mcmc:
        raise SystemExit("Run --mnpe and --mcmc first; missing timings.")

    rows = []
    for budget, r in mnpe["budgets"].items():
        rows.append({
            "component": f"MNPE one-off (budget {int(budget):,})",
            "time_s": r["sim_time_s"] + float(np.mean(r["train_times_s"])),
            "detail": (
                f"sim {r['sim_time_s']:.0f}s + train "
                f"{np.mean(r['train_times_s']):.0f}±{np.std(r['train_times_s']):.0f}s "
                f"({len(r['train_times_s'])} reps)"
            ),
        })
    all_sample_times = [t for r in mnpe["budgets"].values() for t in r["sample_times_s"]]
    mnpe_per_obs = float(np.mean(all_sample_times))
    rows.append({
        "component": f"MNPE per obs ({mnpe['n_posterior_samples']:,} samples)",
        "time_s": mnpe_per_obs,
        "detail": f"mean over {len(all_sample_times)} reps",
    })

    for setting, r in mcmc.items():
        rows.append({
            "component": f"MCMC/SMC per obs ({setting})",
            "time_s": r["wall_time_s"],
            "detail": (
                f"{r.get('runs', 1)} run(s) × {r['chains']} chains × "
                f"{r['draws']:,} draws, 25 configs marginalized, "
                f"max R-hat {r.get('max_rhat', float('nan')):.4f}"
            ),
        })

    # Quadrature is the non-MCMC alternative; included for context, not as the
    # headline comparator.

    reference_file = ARTIFACTS_DIR / "deterministic_reference.npz"
    if reference_file.exists():
        reference = np.load(reference_file)
        rows.append({
            "component": "Quadrature per obs (exact)",
            "time_s": float(reference["elapsed_seconds"]),
            "detail": (
                f"{len(reference['gamma']):,} independent draws, exact "
                f"25-config quadrature, rel. error "
                f"{float(reference['quadrature_relative_error']):.1e}"
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "runtime_comparison.csv", index=False)
    print(df.to_string(index=False))

    print("\n── Break-even (observations until MNPE total cost < alternative) ──")
    alternatives = {
        f"MCMC {setting}": m["wall_time_s"] for setting, m in mcmc.items()
    }
    if reference_file.exists():
        alternatives["quadrature"] = float(
            np.load(reference_file)["elapsed_seconds"]
        )
    for budget, r in mnpe["budgets"].items():
        one_off = r["sim_time_s"] + float(np.mean(r["train_times_s"]))
        for label, per_obs in alternatives.items():
            n_star = one_off / (per_obs - mnpe_per_obs)
            print(f"  budget {int(budget):>7,} vs {label:>16}: n* = {n_star:.1f}")

    print(
        "\nScope. These are costs for THIS problem: three continuous dimensions\n"
        "and 25 configurations summed exactly per density evaluation. SMC is\n"
        "cheap here for those reasons, so amortization pays off only in the\n"
        "many-observation regime. The general argument against the MCMC route\n"
        "is that exact marginalization costs K^D likelihood evaluations per\n"
        "density evaluation while MNPE's categorical head costs one softmax\n"
        "per discrete variable; the scaling experiments, not this benchmark,\n"
        "are the evidence for that."
    )

    hw = results.get("hardware", {})
    print(f"\nHardware: {hw.get('cpu')} ({hw.get('cores')} cores), {hw.get('platform')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mnpe", action="store_true")
    p.add_argument("--mcmc", choices=list(MCMC_SETTINGS))
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    if args.mnpe:
        time_mnpe()
    if args.mcmc:
        time_mcmc(args.mcmc)
    if args.report:
        report()
    if not (args.mnpe or args.mcmc or args.report):
        p.print_help()
