# read python dict back from the file
import os
import pickle

import matplotlib.pyplot as plt
import torch

from botorch.test_functions.multi_objective import C2DTLZ2

Function = {"C2DTLZ2": C2DTLZ2}
methods = ["macKG"]  # ["DISCKG", "HYBRIDKG", "MCKG", "ONESHOTKG"]

performance_comparison = {}
for f in Function.keys():
    function_class = Function[f]

    stats_methods = {}
    stats_methods_parameters = {}
    for i, m in enumerate(methods):
        experiment_path = "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/C2DTLZ2_experiments/" + f + "/" + m +"/"

        discretisation_sizes = os.listdir(experiment_path)
        stats_disc_size = {}
        stats_parameters = {}
        for dsize in discretisation_sizes:
            seed_path = experiment_path + dsize + "/"
            seed_file = os.listdir(seed_path)
            seed_file.sort()
            KGVALS = []

            for sf in seed_file:
                pkl_file = open(seed_path + sf, "rb")
                dict = pickle.load(pkl_file)
                pkl_file.close()
                kgval = dict["acq_outputs"][0]
                num_fantasies = dict["number_of_fantasies"]
                num_X_discrete = dict["number_of_discrete_points"]
                num_scalarizations = dict["num_scalarizations"]
                KGVALS.append(kgval)

            stats_disc_size[dsize] = torch.Tensor(KGVALS).squeeze()
            stats_parameters[dsize] = torch.Tensor([num_fantasies, num_X_discrete, num_scalarizations])

        stats_methods[m] = stats_disc_size
        stats_methods_parameters[m] = stats_parameters

print(stats_methods)
print(stats_methods_parameters)
raise
best_performance = stats_methods["DISCKG"]['8000']

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
        if len(mc_values) < len(best_performance):
            OC = torch.sum(torch.abs((best_performance[:len(mc_values)] - mc_values)))
        else:
            OC = torch.sum(torch.abs(best_performance - mc_values[:len(best_performance)]))
        OCvals.append(OC)

    plt.scatter(np.array(X_plot).reshape(-1), np.array(OCvals).reshape(-1))
    plt.yscale("log")
    plt.show()
