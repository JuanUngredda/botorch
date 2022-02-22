import logging
import os
import pickle as pkl
import subprocess as sp
import sys
from itertools import product

import torch
from botorch.test_functions.multi_objective import C2DTLZ2
from config import CONFIG_DICT
from optimizers.optimizer import Optimizer
from optimizers.utils import KG_wrapper


logger = logging.getLogger(__name__)

GIT_ROOT = sp.check_output(["git", "rev-parse", "--show-toplevel"]).decode()[:-1]

# set path to use copy of source
sys.path.insert(0, GIT_ROOT + "/experiment_scripts/")

HOSTNAME = sp.check_output(["hostname"], shell=True).decode()[:-1]

# set directory to save files
script_dir = os.path.dirname(os.path.abspath(__file__))

dtype = torch.double


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

    # print out all the passed arguments
    for k, v in locals().items():
        logger.info("Printing arguments")
        print(k, ":\t", v)

    # instantiate the test problem
    testfun_dict = {
        "C2DTLZ2": C2DTLZ2,
    }

    CONFIG_NUMBER_FANTASIES = CONFIG_DICT[experiment_name]["num_fantasies"][
        experiment_tag
    ]
    # TODO: Change objective function parametrisation to not be hard-coded
    d = 12
    M = 2

    testfun = testfun_dict[problem](dim=d, num_objectives=M, negate=True).to(dtype=dtype)
    dim = testfun.dim
    bounds = testfun.bounds  # Bounds tensor (2, d)
    lb, ub = bounds
    testfun.problem = problem
    bounds_normalized = torch.vstack([torch.zeros(dim), torch.ones(dim)])

    CONFIG_NUMBER_FANTASIES = CONFIG_DICT[experiment_name]["num_fantasies"][
        experiment_tag
    ]
    CONFIG_NUMBER_DISCRETE_POINTS = CONFIG_DICT[experiment_name]["num_discrete_points"][
        experiment_tag
    ]
    CONFIG_NUMBER_RESTARTS_INNER_OPT = CONFIG_DICT[experiment_name][
        "num_restarts_inner_optimizer"
    ][experiment_tag]
    CONFIG_NUMBER_RAW_SAMPLES_INNER_OPT = CONFIG_DICT[experiment_name][
        "raw_samples_inner_optimizer"
    ][experiment_tag]

    CONFIG_ACQ_OPTIMIZER = CONFIG_DICT[experiment_name]["acquisition_optimizer"][
        experiment_tag
    ]

    CONFIG_NUMBER_RESTARTS_ACQ_OPT = CONFIG_DICT[experiment_name][
        "num_restarts_acq_optimizer"
    ][experiment_tag]
    CONFIG_NUMBER_RAW_SAMPLES_ACQ_OPT = CONFIG_DICT[experiment_name][
        "raw_samples_acq_optimizer"
    ][experiment_tag]

    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

    acquisition_function = KG_wrapper(
        method=method,
        bounds=bounds_normalized,
        num_fantasies=CONFIG_NUMBER_FANTASIES,
        num_discrete_points=CONFIG_NUMBER_DISCRETE_POINTS,
        num_restarts=CONFIG_NUMBER_RESTARTS_INNER_OPT,
        raw_samples=CONFIG_NUMBER_RAW_SAMPLES_INNER_OPT,
    )
    # instantiate the optimizer

    optimizer = Optimizer(
        testfun=testfun,
        acquisitionfun=acquisition_function,
        lb=lb,
        ub=ub,
        n_init=10,  # n_init,
        n_max=50,  # n_max,
        kernel_str="Matern",
        save_folder=savefile,
        base_seed=base_seed,
        optional={
            "NOISE_OBJECTIVE": False,
            "OPTIMIZER": CONFIG_ACQ_OPTIMIZER,
            "RAW_SAMPLES": CONFIG_NUMBER_RAW_SAMPLES_ACQ_OPT,
            "NUM_RESTARTS": CONFIG_NUMBER_RESTARTS_ACQ_OPT,
        },
    )

    # optmize the test problem
    optimizer.optimize()

    # save the output
    output = {
        "problem": problem,
        "method": method,
        "number_of_fantasies": CONFIG_NUMBER_FANTASIES,
        "number_of_discrete_points": CONFIG_NUMBER_DISCRETE_POINTS,
        "number_of_restarts_inner_optimizer": CONFIG_NUMBER_RESTARTS_INNER_OPT,
        "number_of_raw_samples_inner_optimizer": CONFIG_NUMBER_RAW_SAMPLES_INNER_OPT,
        "number_of_restarts_acq_optimizer": CONFIG_NUMBER_RESTARTS_ACQ_OPT,
        "number_of_raw_samples_acq_optimizer": CONFIG_NUMBER_RAW_SAMPLES_ACQ_OPT,
        "base_seed": base_seed,
        "acquisition_optimizer": CONFIG_ACQ_OPTIMIZER,
        "kernel": optimizer.kernel_name,
        "gp_lik_noise": optimizer.gp_likelihood_noise,
        "gp_lengthscales": optimizer.gp_lengthscales,
        "method_times": optimizer.method_time,
        "OC": optimizer.performance,
        "x": optimizer.x_train,
        "y": optimizer.y_train,
        "cwd": os.getcwd(),
        "savefile": savefile,
        "HOSTNAME": HOSTNAME,
        "GIT_ROOT": GIT_ROOT,
    }

    if os.path.isdir(savefile) == False:
        os.makedirs(savefile)

    with open(savefile + "/" + str(base_seed) + ".pkl", "wb") as f:
        pkl.dump(output, f)


def main(exp_names, seed):
    # make table of experiment settings
    EXPERIMENT_NAME = exp_names
    PROBLEMS = CONFIG_DICT[EXPERIMENT_NAME]["problems"]
    ALGOS = CONFIG_DICT[EXPERIMENT_NAME]["method"]
    EXPERIMENTS = list(product(*[PROBLEMS, ALGOS]))
    logger.info(f"Running experiment: {seed} of {len(EXPERIMENTS)}")

    # run that badboy
    for idx, _ in enumerate(EXPERIMENTS):
        run_experiment(
            experiment_name=EXPERIMENT_NAME,
            experiment_tag=idx,
            problem=EXPERIMENTS[idx][0],
            method=EXPERIMENTS[idx][1],
            savefile=script_dir
            + "/results/"
            + EXPERIMENTS[idx][0]
            + "/"
            + EXPERIMENTS[idx][1],
            base_seed=seed,
        )


if __name__ == "__main__":
    main(sys.argv[1:])

    # parser = argparse.ArgumentParser(description="Run KG experiment")
    # parser.add_argument("--seed", type=int, help="base seed", default=0)
    # parser.add_argument("--exp_name", type=str, help="Experiment name in config file")
    # args = parser.parse_args()
