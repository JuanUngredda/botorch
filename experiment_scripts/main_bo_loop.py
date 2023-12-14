import logging
import os
import subprocess as sp
import sys
import time
import warnings

import gpytorch
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from botorch import fit_gpytorch_mll
from botorch.acquisition import MCKnowledgeGradient
from botorch.acquisition.analytic import PosteriorMean
from botorch.generation.gen import gen_candidates_scipy, get_best_candidates
from botorch.models import FixedNoiseGP
from botorch.optim.initializers import gen_value_function_initial_conditions
from config_computational_comparison import CONFIG_DICT
from experiment_scripts.optimizers.test_functions.gp_synthetic_test_function import GP_synthetic
from optimizers.utils import KG_wrapper
from optimizers.utils import lhc
from utils.acquisition_optimizer_utils import OptimizeAcqfAndGetDesign
from utils.parameter_setter import parameter_setter

logger = logging.getLogger(__name__)

GIT_ROOT = sp.check_output(["git", "rev-parse", "--show-toplevel"]).decode()[:-1]

# set path to use copy of source
sys.path.insert(0, GIT_ROOT + "/experiment_scripts/")

HOSTNAME = sp.check_output(["hostname"], shell=True).decode()[:-1]

# set directory to save files
script_dir = os.path.dirname(os.path.abspath(__file__))

dtype = torch.double


def get_config_labels(optimization_parameters: dict):
    return sorted(optimization_parameters.keys())


def evaluate_true_kg(x_new, model, bounds, x_best):
    start = time.time()
    kg_mc_acquisition_function = MCKnowledgeGradient(
        model,
        bounds=bounds,
        num_fantasies=5,
        num_restarts=80,
        raw_samples=124,
        current_optimiser=x_best
    )

    kg_value = kg_mc_acquisition_function(x_new[:, None, :])
    stop = time.time()
    print("evaluation time...", stop - start)
    return kg_value


def update_model(x_train, y_train, covar_module):
    noise_constraint = Interval(0, 1.0000e-4)

    lengtscale_raw = covar_module.base_kernel.lengthscale.squeeze().detach()
    dim = covar_module.base_kernel.ard_num_dims
    e_ = 1.0000e-4
    lengthscale_constraints = Interval(lengtscale_raw - e_, lengtscale_raw + e_)

    base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraints, ard_num_dims=dim)
    base_kernel._set_lengthscale(lengtscale_raw)
    covar_module = ScaleKernel(
        base_kernel,
    )
    model = FixedNoiseGP(x_train,
                         y_train,
                         torch.full_like(y_train, 1e-5),
                         covar_module=covar_module).to(torch.float64)
    likelihood = FixedNoiseGaussianLikelihood(noise=torch.full_like(y_train.squeeze(), 1e-5),
                                              noise_constraint=noise_constraint)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def knowledge_gradient_bo_loop(experiment_name: str,
                               configuration_design: torch.Tensor,
                               configuration_labels: list,
                               seed: int) -> torch.Tensor:
    """
    ARGS:
        :param experiment_name:
        :param configuration_design:
    """

    # instantiate the test problem
    warnings.filterwarnings("ignore")
    CONFIG_INPUT_DIM = CONFIG_DICT[experiment_name]["num_input_dim"]
    CONFIG_LENGTHSCALE = CONFIG_DICT[experiment_name]["lengthscale"]

    testfn = GP_synthetic(dim=CONFIG_INPUT_DIM,
                          seed=seed,
                          kernel_str="RBF",
                          hypers_ls=CONFIG_LENGTHSCALE,
                          negate=True).to(dtype=dtype)

    dim = testfn.dim
    bounds = torch.vstack([torch.zeros(dim), torch.ones(dim)])

    # -------------------------------------------------------------------------
    # PULL CONFIG VARIABLE VALUES
    optimization_parameters = {}
    for idx, label in enumerate(configuration_labels):
        optimization_parameters[label] = configuration_design[idx]
    labels = optimization_parameters.keys()

    setter = parameter_setter(experiment_name=experiment_name)
    setter.build(optimization_parameters, labels)
    # ---------------------------------------------------------------------------------

    # Generate Initial Data
    config_number_initial_design = CONFIG_DICT[experiment_name]["num_samples_initial_design"]
    config_number_dimensions = CONFIG_DICT[experiment_name]["num_input_dim"]
    covar_module = testfn.covar_module
    x_train = lhc(config_number_initial_design, dim=config_number_dimensions)
    y_train = torch.Tensor([testfn(x_i) for x_i in x_train]).reshape((x_train.shape[0], 1)).to(torch.float64)

    # Fit model
    model = update_model(x_train=x_train, y_train=y_train, covar_module=covar_module)

    # This wrapper includes the parameters to each acquisition function.
    acquisition_function = KG_wrapper(
        method=experiment_name,
        bounds=bounds,
        num_fantasies=setter.get_number_fantasies(),
        num_discrete_points=setter.get_number_discrete_points(),
        num_restarts=setter.get_number_restarts_inner_opt(),
        raw_samples=setter.get_number_raw_samples_inner_opt(),
    )

    x_best, y_best = get_best_pm_location(dim, model)

    kg_approximation = acquisition_function(model=model,
                                            x_best=x_best,
                                            fn_best=y_best)

    Tmax = CONFIG_DICT[experiment_name]["Tmax"]
    optimizer = OptimizeAcqfAndGetDesign(maxtime_sec=Tmax, bounds=bounds, acq_func=kg_approximation)
    x_new, _ = optimizer.optimize()

    y_new = evaluate_true_kg(x_new=x_new, model=model, bounds=bounds, x_best=x_best)

    return y_new


def get_best_pm_location(dim, model):
    value_function = PosteriorMean(model=model)
    x_initial_conditions = gen_value_function_initial_conditions(
        acq_function=value_function,
        bounds=torch.Tensor([[0] * dim, [1] * dim]),
        current_model=model,
        num_restarts=80,  # self.num_restarts,
        raw_samples=1024
    )
    x_pm, best_pm_value = gen_candidates_scipy(
        initial_conditions=x_initial_conditions,
        acquisition_function=value_function,
        lower_bounds=torch.Tensor([0] * dim),
        upper_bounds=torch.Tensor([1] * dim)
    )

    best_candidate = get_best_candidates(x_pm, best_pm_value)
    return best_candidate, value_function(best_candidate)
