import argparse
import logging
import os
import pickle as pkl
import subprocess as sp
import sys
from itertools import product

import torch

from experiment_scripts.config import CONFIG_DICT
from optimizers.optimizer import Optimizer
from optimizers.utils import KG_wrapper
from problems.experiments2d import (eggholder,
                                    powers,
                                    branin,
                                    cosines,
                                    mccormick,
                                    goldstein,
                                    sixhumpcamel,
                                    dropwave,
                                    rosenbrock,
                                    beale)

logger = logging.getLogger(__name__)

GIT_ROOT = sp.check_output(["git", "rev-parse", "--show-toplevel"]).decode()[:-1]

# set path to use copy of source
sys.path.insert(0, GIT_ROOT + "/experiment_scripts/")

HOSTNAME = sp.check_output(["hostname"], shell=True).decode()[:-1]

# set directory to save files
script_dir = os.path.dirname(os.path.abspath(__file__))


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
    testfun_dict = {"Egg-holder": eggholder,
                    "Sum of Powers": powers,
                    "Branin": branin,
                    "Cosines": cosines,
                    "Mccormick": mccormick,
                    "Goldstein": goldstein,
                    "Six-hump camel": sixhumpcamel,
                    "dropwave": dropwave,
                    "Rosenbrock": rosenbrock,
                    "beale": beale
                    }

    testfun = testfun_dict[problem](base_seed=base_seed)

    bounds = torch.vstack([testfun.lb, testfun.ub])  # Bounds tensor (2, d)

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

    acquisition_function = KG_wrapper(
        method=method,
        bounds=bounds,
        num_fantasies=CONFIG_NUMBER_FANTASIES,
        num_discrete_points=CONFIG_NUMBER_DISCRETE_POINTS,
        num_restarts=CONFIG_NUMBER_RESTARTS_INNER_OPT,
        raw_samples=CONFIG_NUMBER_RAW_SAMPLES_INNER_OPT,
    )
    # instantiate the optimizer

    optimizer = Optimizer(
        testfun=testfun,
        acquisitionfun=acquisition_function,
        lb=testfun.lb,
        ub=testfun.ub,
        n_init=10,  # n_init,
        n_max=11,  # n_max,
        kernel_str="Matern",
        save_folder=savefile,
        base_seed=base_seed,
        optional={
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

    with open(savefile + "/" + str(args.seed) + ".pkl", "wb") as f:
        pkl.dump(output, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run KG experiment")
    parser.add_argument("--seed", type=int, help="base seed", default=0)
    args = parser.parse_args()

    # make table of experiment settings
    EXPERIMENT_NAME = "rosenbrock_experiments"
    PROBLEMS = CONFIG_DICT[EXPERIMENT_NAME]["problems"]
    ALGOS = CONFIG_DICT[EXPERIMENT_NAME]["method"]
    EXPERIMENTS = list(product(*[PROBLEMS, ALGOS]))
    logger.info(f"Running experiment: {args.seed} of {len(EXPERIMENTS)}")

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
            base_seed=args.seed,
        )
