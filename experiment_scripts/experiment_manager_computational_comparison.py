import logging
import os
import subprocess as sp
import sys
import time
from itertools import product

import torch
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor

from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models import FixedNoiseGP
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from botorch.utils.transforms import unnormalize
from config_computational_comparison import CONFIG_DICT, true_underlying_config_params
from experiment_scripts.optimizers.test_functions.gp_synthetic_test_function import GP_synthetic
from optimizers.utils import (KG_Objective_Function)
from optimizers.utils import lhc

logger = logging.getLogger(__name__)

GIT_ROOT = sp.check_output(["git", "rev-parse", "--show-toplevel"]).decode()[:-1]

# set path to use copy of source
sys.path.insert(0, GIT_ROOT + "/experiment_scripts/")

HOSTNAME = sp.check_output(["hostname"], shell=True).decode()[:-1]

# set directory to save files
script_dir = os.path.dirname(os.path.abspath(__file__))

dtype = torch.double

NOISE_SE = 1e-4
train_yvar = torch.tensor(NOISE_SE ** 2, dtype=dtype)

def evaluate_test_func(test_fun, x: Tensor, bounds) -> Tensor:
    # bring x \in [0,1]^d to original bounds.
    x = unnormalize(X=x, bounds=bounds)
    y = torch.Tensor([test_fun(x)]).to(dtype=torch.double)
    return y

def generate_model(test_fun, noise, X_train: Tensor, Y_train: Tensor):
    covar_module = test_fun.covar_module
    Y_train_standarized = standardize(Y_train)
    # We can specify that it's deterministic and adding some small noise for numerical stability.
    NOISE_VAR = noise ** 2

    updated_model = FixedNoiseGP(
        train_X=X_train,
        train_Y=Y_train_standarized,
        covar_module=covar_module,
        train_Yvar=NOISE_VAR.expand_as(Y_train_standarized),
    )
    return updated_model

def generate_objective_function(test_fun, x_train, y_train, kg_method, optimiser_bounds, noise,
                                x_optimiser,
                                current_value):
    generate_gp_model = generate_model(test_fun=test_fun, X_train=x_train, Y_train=y_train, noise=noise)

    # This wrapper includes the parameters to each acquisition function.
    acquisition_function = KG_Objective_Function(model=generate_gp_model,
                                                 method=kg_method,
                                                 bounds=optimiser_bounds,
                                                 x_optimiser=x_optimiser,
                                                 current_value=current_value)

    return acquisition_function

def generate_desings(n, dim):
    x_train = lhc(n, dim=dim)  # Tensor (n_init , X_dim)
    return x_train

def rescaling(x, bounds):
    lb, ub = bounds
    return x * (ub - lb) + lb

def adapting_type(x, type):
    return x.type(type)

def generate_initial_data(objective_fun, dim, config_parameters, n=10):
    types = [config_parameters[k][1] for k in config_parameters]
    bounds = [config_parameters[k][0] for k in config_parameters]
    assert dim == len(config_parameters);
    "check dims"
    # generate training data
    train_x = []
    X = generate_desings(n=n, dim=dim).T
    for idx, x in enumerate(X):
        x_rescaled = rescaling(x, bounds=bounds[idx])
        x_type_adapted = adapting_type(x_rescaled, type=types[idx])
        train_x.append(x_type_adapted)

    train_x = torch.vstack(train_x).T
    train_obj = torch.Tensor([objective_fun(x_i) for x_i in train_x]).unsqueeze(-1)  # add output dimension
    best_observed_value = train_obj.max().item()
    return train_x, train_obj, best_observed_value

def update_model(train_x, train_obj, state_dict=None):
    # define models for objective and constraint
    model = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    # combine into a multi-output GP model
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

class optimize_acqf_and_get_design():

    def __init__(self, maxtime_sec, acq_func, bounds, **kwargs):
        self.nit = 0
        self.maxtime_sec = maxtime_sec
        self.acq_func = acq_func
        self.bounds = bounds

        self.raw_samples = int(kwargs["raw_samples_external_optimizer"])
        self.num_restarts = int(max(self.raw_samples * float(kwargs["proportion_restarts_external_optimizer"]), 1))

    def fun(self, x_val):
        y_val = self.acq_func(x_val)
        self.x.append(x_val)
        self.y.append(y_val)
        print("self.x", x_val)
        print("self.y", y_val)
        return y_val

    def get_best_x(self):
        y_tensor = torch.Tensor(self.y)
        return self.x[torch.argmax(y_tensor)]

    def get_best_y(self):
        y_tensor = torch.Tensor(self.y)
        return torch.max(y_tensor)

    def callback(self, x):
        # callback to terminate if maxtime_sec is exceeded
        self.nit += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.maxtime_sec:
            print("Elapsed: %.3f sec" % elapsed_time)
            raise Exception("Terminating optimization: time limit reached")
        else:
            # you could print elapsed iterations and time
            print("Elapsed: %.3f sec" % elapsed_time)
            print("Elapsed iterations: ", self.nit)

    def optimize(self):
        self.start_time = time.time()
        self.x = []
        self.y = []

        try:
            candidates = optimize_acqf(
                acq_function=self.fun,
                bounds=self.bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"callback": self.callback},
                return_best_only=True
            )
        except:
            pass
        stop_time = time.time()
        print("self.start_time", self.maxtime_sec)
        print("self.bounds", self.bounds)
        print("self.num_restarts", self.num_restarts)
        print("self.raw_samples", self.raw_samples)
        # print("best_x", self.get_best_x())
        # print("best_y", self.get_best_y())
        # print("self.maxtime_sec",self.maxtime_sec)
        # print("dif time", stop_time - self.start_time)
        raise
        return candidates

def get_optimisation_bounds(dict_vars):
    lb = torch.Tensor([0 for _ in range(len(dict_vars))])
    ub = torch.Tensor([0 for _ in range(len(dict_vars))])
    for idx, var_key in enumerate(dict_vars):
        bounds = dict_vars[var_key][0]
        lb[idx], ub[idx] = bounds[0], bounds[1]
    tensor_bounds = torch.vstack((lb, ub))
    return tensor_bounds

def objective_function_constructor(test_fun, test_fun_bounds, keys, method, X_TRAIN_GP_MODEL, Y_TRAIN_GP_MODEL, tmax,
                                   optimisation_bounds):
    objective_function = generate_objective_function(test_fun=test_fun,
                                                     optimiser_bounds=test_fun_bounds,
                                                     kg_method=method,
                                                     x_train=X_TRAIN_GP_MODEL,
                                                     y_train=Y_TRAIN_GP_MODEL,
                                                     x_optimiser=None,
                                                     current_value=None,
                                                     noise=torch.Tensor([1e-3]))

    underlying_objective_func = generate_objective_function(test_fun=test_fun,
                                                            optimiser_bounds=test_fun_bounds,
                                                            kg_method="MCKG",
                                                            x_train=X_TRAIN_GP_MODEL,
                                                            y_train=Y_TRAIN_GP_MODEL,
                                                            x_optimiser=None,
                                                            current_value=None,
                                                            noise=torch.Tensor([1e-3]))(**true_underlying_config_params)

    def objective_function_evaluator(x):
        input_dictionary = {k: x[idx] for idx, k in enumerate(keys)}
        opt = optimize_acqf_and_get_design(acq_func=objective_function(**input_dictionary),
                                           maxtime_sec=tmax,
                                           bounds=test_fun_bounds,
                                           **input_dictionary)
        candidates, _ = opt.optimize()
        x_new = candidates
        y_new = underlying_objective_func(torch.atleast_2d(x_new))
        return y_new

    return objective_function_evaluator

def run_experiment(
        experiment_name: str,
        experiment_tag: int,
        problem: str,
        method: str,
        savefile: str,
        base_seed: int,
        n_init=4,
        n_max=50,
):
    """
    ARGS:
        problem: str, a key from the dict of problems
        method: str, name of optimizer
        savefile: str, locaiotn to write output results
        base_seed: int, generates testfun
        n_init: int, starting budfget of optimizer
        n_max: ending budget for optimizer
    """

    # Problem Parameters
    INPUT_DIM = CONFIG_DICT[experiment_name]["num_input_dim"][experiment_tag]
    LENGTHSCALE = CONFIG_DICT[experiment_name]["lengthscale"][experiment_tag]
    ACQ_OPTIMIZER = CONFIG_DICT[experiment_name]["acquisition_optimizer"][experiment_tag]
    CONFIG_TMAX = CONFIG_DICT[experiment_name]["Tmax"][experiment_tag]
    CONFIG_NUMBER_INITAL_DESIGN = CONFIG_DICT[experiment_name]["num_samples_initial_design"][experiment_tag]
    CONFIG_PARAMETERS = CONFIG_DICT[experiment_name]["optimization_parameters"]

    # instantiate GP function
    testfun_dict = {
        "GP_synthetic": GP_synthetic
    }
    testfun = testfun_dict[problem](dim=INPUT_DIM,
                                    seed=base_seed,
                                    hypers_ls=LENGTHSCALE,
                                    negate=True).to(dtype=dtype)

    testfun_dim = testfun.dim
    testfun_bounds = testfun.bounds  # Bounds tensor (2, d)
    lb, ub = testfun_bounds

    optimiser_bounds = get_optimisation_bounds(CONFIG_PARAMETERS)
    optimiser_dim = len(CONFIG_PARAMETERS)

    testfun.problem = problem
    bounds_normalized = torch.vstack([torch.zeros(optimiser_dim), torch.ones(optimiser_dim)])

    # generate objective function:
    X_TRAIN_GP_MODEL = generate_desings(n=CONFIG_NUMBER_INITAL_DESIGN, dim=testfun_dim)
    Y_TRAIN_GP_MODEL = torch.Tensor(
        [evaluate_test_func(test_fun=testfun, x=x_i, bounds=testfun_bounds) for x_i in X_TRAIN_GP_MODEL]
    ).reshape((X_TRAIN_GP_MODEL.shape[0], 1))  # Tensor (n_init , 1)

    objective_function = objective_function_constructor(test_fun=testfun,
                                                        method=method,
                                                        test_fun_bounds=testfun_bounds,
                                                        X_TRAIN_GP_MODEL=X_TRAIN_GP_MODEL,
                                                        Y_TRAIN_GP_MODEL=Y_TRAIN_GP_MODEL,
                                                        keys=CONFIG_PARAMETERS.keys(),
                                                        tmax=CONFIG_TMAX,
                                                        optimisation_bounds=optimiser_bounds)

    train_x_ei, train_obj_ei, best_observed_value_ei = generate_initial_data(dim=optimiser_dim,
                                                                             config_parameters=CONFIG_PARAMETERS,
                                                                             objective_fun=objective_function,
                                                                             n=CONFIG_NUMBER_INITAL_DESIGN)
    raise
    verbose = False
    best_observed_ei, best_random_all = [], []
    for iteration in range(1, N_BATCH + 1):
        t0 = time.monotonic()
        # Generate acquisition function
        # for best_f, we use the best observed noisy values as an approximation

        mll_ei, model_ei = update_model(train_x_ei, train_obj_ei)
        EI = ExpectedImprovement(
            model=model_ei,
            best_f=train_obj_ei.max(),
            maximize=True
        )

        # optimize and get new observation
        new_x_ei, = optimize_acqf_and_get_design(EI)
        new_x_random = generate_desings(n=1)
        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_y_ei = torch.cat([train_y_ei, untimed_objective_function(new_x_ei)])

        best_observed_ei.append(train_y_ei)
        t1 = time.monotonic()
        if verbose:
            y_true_ei = underlying_objective_func(new_x_ei)
            y_true_random = underlying_objective_func(new_x_random)
            # print(
            #     f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI) = "
            #     f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}), "
            #     f"time = {t1 - t0:>4.2f}.", end=""
            # )
        else:
            print(".", end="")


def main(exp_names, seed):
    # make table of experiment settings
    EXPERIMENT_NAME = exp_names
    PROBLEMS = CONFIG_DICT[EXPERIMENT_NAME]["problems"]
    ALGOS = CONFIG_DICT[EXPERIMENT_NAME]["method"]
    EXPERIMENTS = list(product(*[PROBLEMS, ALGOS]))
    logger.info(f"Running experiment: {seed} of {len(EXPERIMENTS)}")

    # run that badboy
    for idx, _ in enumerate(EXPERIMENTS):

        file_name = script_dir + "/results/" + EXPERIMENT_NAME + "/" + EXPERIMENTS[idx][0] + "/" + EXPERIMENTS[idx][
            1] + "/" + str(seed) + ".pkl"

        if os.path.isfile(file_name) == False:
            run_experiment(
                experiment_name=EXPERIMENT_NAME,
                experiment_tag=idx,
                problem=EXPERIMENTS[idx][0],
                method=EXPERIMENTS[idx][1],
                savefile=script_dir + "/results/" + EXPERIMENT_NAME + "/" + EXPERIMENTS[idx][0] + "/" +
                         EXPERIMENTS[idx][
                             1],
                base_seed=seed,
            )


if __name__ == "__main__":
    main(sys.argv[1:])

    # parser = argparse.ArgumentParser(description="Run KG experiment")
    # parser.add_argument("--seed", type=int, help="base seed", default=0)
    # parser.add_argument("--exp_name", type=str, help="Experiment name in config file")
    # args = parser.parse_args()
