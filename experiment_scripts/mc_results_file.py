import pickle

import matplotlib.pyplot as plt
import torch
import os

# read python dict back from the file
from botorch.test_functions import EggHolder, Branin, SixHumpCamel, Rosenbrock, Hartmann
from experiment_scripts.optimizers.test_functions.gp_synthetic_test_function import GP_synthetic

Function = { "GP_synthetic": GP_synthetic}
methods = ["DISCKG"]  # ["DISCKG", "HYBRIDKG", "MCKG", "ONESHOTKG"]

performance_comparison = {}
for f in Function.keys():
    function_class = Function[f]

    stats_methods = {}
    for i, m in enumerate(methods):
        experiment_path = "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/"+m +"_"+f+"_dim2_l0.1/"+f+"/"+ m+"/"
        discretisation_sizes = os.listdir(experiment_path)
        discretisation_sizes.remove("0")

        stats_disc_size = {}
        for dsize in discretisation_sizes:
            seed_path = experiment_path +dsize +"/"
            seed_file = os.listdir(seed_path)
            seed_file.sort()
            KGVALS = []
            for sf in seed_file:
                pkl_file = open(seed_path+sf, "rb")
                dict = pickle.load(pkl_file)
                pkl_file.close()
                if len( dict["acq_outputs"])!=0:
                    kgval = dict["acq_outputs"][0]
                    KGVALS.append(kgval)

            stats_disc_size[dsize] = torch.Tensor(KGVALS).squeeze()
        stats_methods[m] = stats_disc_size

best_performance = stats_methods["DISCKG"]['5000']

import numpy as np
for i, m in enumerate(methods):
    discretisation_sizes = stats_methods[m].keys()
    OCvals = []
    X_plot = []
    for dsize in discretisation_sizes:
        # if dsize != "5000":
        X_plot.append(int(dsize))
        mc_values = stats_methods[m][dsize]
        # OC = torch.mean(mc_values)
        if len(mc_values)< len(best_performance):
            OC = torch.sum(torch.abs((best_performance[:len(mc_values)] - mc_values)))
        else:
            OC = torch.sum(torch.abs(best_performance - mc_values[:len(best_performance)]))
        OCvals.append(OC)

    plt.scatter(np.array(X_plot).reshape(-1), np.array(OCvals).reshape(-1))
    # plt.yscale("log")
    plt.show()