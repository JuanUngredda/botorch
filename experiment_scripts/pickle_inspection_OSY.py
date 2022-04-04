import pickle

import matplotlib.pyplot as plt
import torch

# read python dict back from the file
from botorch.test_functions.multi_objective import OSY

methods = ["macKG"]


for i, m in enumerate(methods):
    pkl_file = open(
        "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/OSY_experiments/OSY/"
        + m
        + "/Tche/0.pkl",
        "rb",
    )
    mydict2 = pickle.load(pkl_file)
    pkl_file.close()

    X = mydict2["x"]
    Y = mydict2["y"].squeeze()
    C = mydict2["c"]
    sampled_is_feas = (C < 0)
    print(sampled_is_feas.shape)
    print("C", C)
    print("sampled_is_feas", sampled_is_feas)
    sampled_aggregated_is_feas = torch.prod(sampled_is_feas, dim=-1, dtype=bool)
    print("sampled_aggregated_is_feas",sampled_aggregated_is_feas)
    print(sampled_aggregated_is_feas.shape)
    print("Y", Y)

    Y_feas = Y[sampled_aggregated_is_feas]
    print("Y_feas", Y_feas)
    # raise
    print(mydict2["method_times"])
    fun = OSY(negate=True)
    d = fun.dim
    bounds = fun.bounds  # Bounds tensor (2, d)
    print(bounds)
    X_plot = torch.rand(100000, d) * (bounds[1] - bounds[0]) + bounds[0]

    Y_plot = fun(X_plot)
    C_plot = fun.evaluate_slack(X_plot)
    is_feas = -C_plot <= 0

    # print(type(is_feas))
    aggregated_is_feas = torch.prod(is_feas, dim=-1 , dtype=bool)

    print(aggregated_is_feas.sum())
    plt.scatter(Y_plot[:, 0], Y_plot[:, 1], color="yellow")
    plt.scatter(Y_plot[aggregated_is_feas,0], Y_plot[aggregated_is_feas,1], color="green", s=5)
    plt.scatter(Y[:, 0], Y[:, 1], color="blue")
    plt.scatter(Y[:14, 0], Y[:14, 1], color="magenta", marker="x")
    plt.scatter(Y_feas[:, 0], Y_feas[:, 1], color="orange")

    plt.xlim((0, 300))
    plt.ylim((-200, 0))
    plt.show()
    raise
    plt.title(m + " performance from GP")
    plt.plot(mydict2["OC_GP"][:, 0], mydict2["OC_GP"][:, 1])
    plt.legend()
    plt.show()

    plt.title(m + " performance from sampled")
    plt.plot(mydict2["OC_sampled"][:, 0], mydict2["OC_sampled"][:, 1])
    plt.legend()
    plt.show()


# pkl_file = open(
#     "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/Branin/HYBRIDKG/0.pkl",
#     "rb",
# )
# mydict2 = pickle.load(pkl_file)
# pkl_file.close()
#
# print(mydict2["method_times"])
