"""The jaxley point-neuron simulator, its summary statistics and the mixed prior.

Four continuous parameters (g_Na, g_K, g_Leak, E_Leak) on a sigmoid-transformed
box, a binary M-type potassium switch and a three-way calcium choice (off, L, T).
"""

import warnings

import jax
import jax.lax as lax
import jax.numpy as jnp
import jaxley as jx

# To suppress Pandas performance warnings:
import pandas as pd
import sbi
import torch
from jaxley.channels import CaL, CaT, K, Km, Leak, Na
from jaxley.optimize.transforms import SigmoidTransform
from sbi.utils import MultipleIndependent

warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)


# define HH model
cell = jx.Cell()  # Define cell.
cell.set("length", 100.0)  # um
cell.set("radius", 100.0)  # um
cell.insert(Na())
cell.insert(K())
cell.insert(Km())
cell.insert(CaL())
cell.insert(CaT())
cell.insert(Leak())
# with stimulus
# Allen protocol stimulation
dt = 0.02  # ms
t_max = 1449.98  # ms

cell.delete_stimuli()
cell.delete_recordings()

A = cell.nodes["area"].item()

i_delay = 205.0  # ms
i_dur = 999.98  # ms
i_amp_norm = 2  # 1.8  # nA
i_amp = i_amp_norm * A / 100_000  # nA

current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
cell.branch(0).loc(0.5).stimulate(current)
cell.branch(0).loc(0.5).record()

cell.set("v", -85.0)
cell.init_states()

# set default parameters for fixed channels
_Km_gKm = 0.03e-3
_CaL_gCaL = 0.1e-3
_CaT_gCaT = 0.4e-3
cell.set("Km_gKm", _Km_gKm)
cell.set("CaL_gCaL", _CaL_gCaL)
cell.set("CaT_gCaT", _CaT_gCaT)


# define priors
HH_n_continous = 4
n_binary = 1
n_categorical = 1
n_categories = 3  # for the categorical variable
# priors for continous parameters
bounds = {}
bounds["gNa"] = [8e-3, 8e-2]  # 0.03
bounds["gK"] = [1.5e-3, 1.5e-2]  # 1e-2
bounds["gLeak"] = [1e-5, 1e-4]  # 1e-4
bounds["ELeak"] = [-80.0, -60.0]  # -72.0
# Extract the lower and upper bounds as an array, for convenience later.
lower_bounds = torch.tensor(list(bounds.values()))[:, 0]
upper_bounds = torch.tensor(list(bounds.values()))[:, 1]

HH_transform = SigmoidTransform(
    lower=jnp.array(lower_bounds),
    upper=jnp.array(upper_bounds),
)

HH_prior = MultipleIndependent(
    [sbi.utils.BoxUniform(low=lower_bounds, high=upper_bounds)]
    + [torch.distributions.bernoulli.Bernoulli(torch.ones(1) * 0.5)] * n_binary
    + [
        torch.distributions.categorical.Categorical(
            probs=torch.ones(1, n_categories) / n_categories
        )
    ]
    * n_categorical
)


def simulate(params, n_continous=4, n_discrete=2, key=jax.random.PRNGKey(123)):
    # takes the normalized parameters as input
    key, subkey = jax.random.split(key)

    params_continous = params[:n_continous]
    params_continous = HH_transform(params_continous)
    params_discrete = params[n_continous:]

    assert len(params_discrete) == n_discrete

    pstate = None

    # Set Km: pstate[0] corresponds to Km_gKm
    pstate = cell.data_set("Km_gKm", 0.03e-3, pstate)
    gKm_cond = lax.cond(
        params_discrete[0] == 0,
        lambda _: 0.0,
        lambda _: _Km_gKm,
        operand=None,
    )
    pstate[0]["val"] = jnp.ones_like(pstate[0]["val"]) * gKm_cond

    # CaL: pstate[1] corresponds to CaL_gCaL
    pstate = cell.data_set("CaL_gCaL", 0.0, pstate)
    gCaL_cond = lax.cond(
        params_discrete[1] == 1,
        lambda _: _CaL_gCaL,
        lambda _: 0.0,
        operand=None,
    )
    pstate[1]["val"] = jnp.ones_like(pstate[1]["val"]) * gCaL_cond

    # CaT: pstate[2] corresponds to CaT_gCaT
    pstate = cell.data_set("CaT_gCaT", 0.0, pstate)
    gCaT_cond = lax.cond(
        params_discrete[1] == 2,
        lambda _: _CaT_gCaT,
        lambda _: 0.0,
        operand=None,
    )
    pstate[2]["val"] = jnp.ones_like(pstate[2]["val"]) * gCaT_cond

    # set continous parameters
    pstate = cell.data_set("Na_gNa", params_continous[0], pstate)
    pstate = cell.data_set("K_gK", params_continous[1], pstate)
    pstate = cell.data_set("Leak_gLeak", params_continous[2], pstate)
    pstate = cell.data_set("Leak_eLeak", params_continous[3], pstate)

    # set initial voltage to leak reversal potential
    pstate = cell.data_set("v", params_continous[3], pstate)

    # Return [0] because the result of `jx.integrate` is of shape (1, time). Here, we
    # get rid of the batch dimension.
    trace = jx.integrate(cell, param_state=pstate)[0]
    noise = jax.random.normal(subkey, trace.shape) * 0.1  # adding noise

    return trace + noise


HH_sim = jax.jit(jax.vmap(simulate, in_axes=(0,)))


def HH_summary_stats(v):
    dt = 0.02  # time step in ms
    # define summary statistics
    window0 = jnp.arange(0, 10250)  # first 205ms before stimulus # Unit: time steps.
    window1 = jnp.arange(10250, 12750)  # first 50ms of stimulus # Unit: time steps.
    window2 = jnp.arange(12750, 20250)  # 50ms to 200ms of stimulus # Unit: time steps.
    window3 = jnp.arange(
        20250, 30250
    )  #  200ms to 400ms of stimulus # Unit: time steps.
    window4 = jnp.arange(30250, 60248)  # 400ms until end # Unit: time steps.

    mean0 = jnp.mean(v[window0])
    max1 = jnp.max(v[window1])
    mean1 = jnp.mean(v[window1])
    std1 = jnp.std(v[window1])
    max2 = jnp.max(v[window2])
    mean2 = jnp.mean(v[window2])
    std2 = jnp.std(v[window2])
    max3 = jnp.max(v[window3])
    mean3 = jnp.mean(v[window3])
    std3 = jnp.std(v[window3])
    max4 = jnp.max(v[window4])
    mean4 = jnp.mean(v[window4])
    std4 = jnp.std(v[window4])
    high_vals = ((v > -10).sum()) * dt  # total time above -10 mV, in ms
    return jnp.asarray(
        [
            mean0,
            max1,
            mean1,
            std1,
            max2,
            mean2,
            std2,
            max3,
            mean3,
            std3,
            max4,
            mean4,
            std4,
            high_vals,
        ]
    )
