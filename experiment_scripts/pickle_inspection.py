import pickle

import matplotlib.pyplot as plt
import torch

# read python dict back from the file
from botorch.test_functions import EggHolder, Branin, SixHumpCamel, Rosenbrock

methods = ["DISCKG", "HYBRIDKG", "MCKG", "ONESHOTKG"]

performance_comparison = {}
for i, m in enumerate(methods):
    pkl_file = open(
        "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/Branin/"
        + m
        + "/0.pkl",
        "rb",
    )
    mydict2 = pickle.load(pkl_file)
    pkl_file.close()

    X = mydict2["x"]
    Y = mydict2["y"]
    f = Branin(negate=True)
    op_value = f.optimal_value
    opt_val = torch.Tensor([[i[0], i[1]] for i in f._optimizers])
    bounds = f.bounds  # Bounds tensor (2, d)
    X_plot = torch.rand(10000, 2) * (bounds[1] - bounds[0]) + bounds[0]
    Y_plot = f(X_plot).squeeze()
    plt.title(m)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=Y_plot)
    plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], label="algorithm iterations")
    plt.scatter(X[:10, 0], X[:10, 1], color="red", label="initial design")
    plt.scatter(opt_val[:, 0], opt_val[:, 1], color="black")
    plt.legend()

    plt.show()

    plt.title(m + " performance from GP")
    plt.plot(mydict2["OC"][:, 0], mydict2["OC"][:, 1])
    plt.legend()
    plt.show()

    plt.title(m + " performance from sampled")
    plt.plot(torch.cummax(Y.squeeze(), dim=0)[0])
    plt.legend()
    plt.show()

    performance_comparison[m] = mydict2["OC"][:, 1]

plt.title("performance comparison")
for i, m in enumerate(methods):
    plt.plot(op_value-performance_comparison[m], label=m)
plt.legend()
plt.yscale("log")
plt.show()
# pkl_file = open(
#     "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/Branin/HYBRIDKG/0.pkl",
#     "rb",
# )
# mydict2 = pickle.load(pkl_file)
# pkl_file.close()
#
# print(mydict2["method_times"])
