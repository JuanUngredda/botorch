import logging
import os
import subprocess as sp
import sys

import gpytorch
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from torch import Tensor

from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models import FixedNoiseGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from config_computational_comparison import CONFIG_DICT
from main_bo_loop import knowledge_gradient_bo_loop
from optimizers.utils import lhc
from utils.save_utils import save_results

logger = logging.getLogger(__name__)

GIT_ROOT = sp.check_output(["git", "rev-parse", "--show-toplevel"]).decode()[:-1]

# set path to use copy of source
sys.path.insert(0, GIT_ROOT + "/experiment_scripts/")

HOSTNAME = sp.check_output(["hostname"], shell=True).decode()[:-1]

# set directory to save files
script_dir = os.path.dirname(os.path.abspath(__file__))

dtype = torch.double
NUM_RESTARTS = 10
RAW_SAMPLES = 512
NOISE_SE = 1e-4
train_yvar = torch.tensor(NOISE_SE ** 2, dtype=dtype)


def main(experiment_name, seed):
    # make table of experiment settings
    algorithms = CONFIG_DICT[experiment_name]["method"]
    for algorithm in algorithms:
        base_path = script_dir + "/results/" + algorithm
        file_name = base_path + "/" + str(seed) + ".pkl"
        if not os.path.isfile(file_name):
            run_experiment(
                experiment_name=experiment_name,
                save_file=base_path,
                base_seed=seed,
            )


def run_experiment(
        experiment_name: str,
        base_seed: int,
        save_file: str,
        n_max=100):
    torch.random.manual_seed(base_seed)
    # Problem Parameters
    config_number_initial_design = CONFIG_DICT[experiment_name]["num_samples_initial_design"]
    config_optimization_parameters = CONFIG_DICT[experiment_name]["optimization_parameters"]

    # get configuration bounds and dimensions
    configuration_space_bounds = get_optimisation_bounds(config_optimization_parameters)
    configuration_space_dimension = configuration_space_bounds.shape[1]
    configuration_space_bounds_normalized = bounds_normalizer(configuration_space_dimension)

    # Generate configuration designs
    x_train = generate_designs(n=config_number_initial_design, dim=configuration_space_dimension)
    y_train = torch.Tensor(
        [evaluate_bo_loop(experiment_name=experiment_name,
                          configuration_design=x_i,
                          configuration_labels=list(config_optimization_parameters.keys()),
                          bounds=configuration_space_bounds,
                          seed=base_seed) for x_i in x_train]).to(torch.float64)
    model = update_model(x_train, y_train)
    storage = save_results(kg_type=experiment_name,
                           keys=list(config_optimization_parameters.keys()),
                           bounds=configuration_space_bounds,
                           seed=base_seed,
                           save_folder=save_file)
    storage.save(model, rescaling(x_train, bounds=configuration_space_bounds), y_train)
    for _ in range(n_max):
        acquisition_function = ExpectedImprovement(model=model,
                                                   best_f=((y_train - y_train.mean()) / y_train.std()).max(),
                                                   maximize=True)
        x_new = get_x_new(acq_func=acquisition_function, bounds=configuration_space_bounds_normalized)
        y_new = torch.Tensor([evaluate_bo_loop(experiment_name=experiment_name,
                                               configuration_design=x_new.reshape(-1),
                                               configuration_labels=list(config_optimization_parameters.keys()),
                                               bounds=configuration_space_bounds,
                                               seed=base_seed)])
        x_train = torch.cat([x_train, x_new])
        y_train = torch.cat([y_train, y_new])
        model = update_model(train_x=x_train, train_obj=y_train)
        storage.save(model, rescaling(x_train, bounds=configuration_space_bounds), y_train)


def get_optimisation_bounds(dict_vars):
    lb = torch.Tensor([0 for _ in range(len(dict_vars))])
    ub = torch.Tensor([0 for _ in range(len(dict_vars))])
    for idx, var_key in enumerate(dict_vars.keys()):
        bounds = dict_vars[var_key][0]
        lb[idx], ub[idx] = bounds[0], bounds[1]
    tensor_bounds = torch.vstack((lb, ub))
    return tensor_bounds


def bounds_normalizer(configuration_space_dimension):
    configuration_space_bounds_normalized = torch.vstack([torch.zeros(configuration_space_dimension),
                                                          torch.ones(configuration_space_dimension)])
    return configuration_space_bounds_normalized


def evaluate_bo_loop(experiment_name: str, configuration_design: torch.Tensor, configuration_labels: list,
                     bounds: torch.Tensor,
                     seed: int) -> Tensor:
    # bring x \in [0,1]^d to original bounds.
    x = unnormalize(X=configuration_design, bounds=bounds)
    return torch.Tensor([knowledge_gradient_bo_loop(experiment_name, x, configuration_labels, seed=seed)]).to(
        dtype=torch.double)


def generate_designs(n, dim):
    x_train = lhc(n, dim=dim)  # Tensor (n_init , X_dim)
    return x_train


def update_model(train_x, train_obj):
    # define models for objective
    standard_y = (train_obj - train_obj.mean()) / train_obj.std()
    train_y_reshape = standard_y.reshape((train_x.shape[0], 1))
    covar_module = ScaleKernel(
        MaternKernel(ard_num_dims=train_x.shape[1]),
    )
    model = FixedNoiseGP(train_x,
                         train_y_reshape,
                         torch.full_like(train_y_reshape, 1e-4).to(torch.float64),
                         covar_module=covar_module
                         )
    # Define the likelihood (negative log likelihood)
    noise_constraint = noise_constraint = Interval(0, 1.0000e-4)
    likelihood = FixedNoiseGaussianLikelihood(noise=torch.full_like(train_obj.squeeze(), 1e-4),
                                              noise_constraint=noise_constraint)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def get_x_new(acq_func, bounds):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
    )
    return candidates.detach()


def rescaling(x, bounds):
    lb, ub = bounds
    return x * (ub - lb) + lb


if __name__ == "__main__":
    main(sys.argv[1:], 0)
