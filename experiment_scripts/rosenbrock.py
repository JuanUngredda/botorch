import time

import matplotlib.pyplot as plt
import torch
from botorch.acquisition import (
    qKnowledgeGradient,
    HybridKnowledgeGradient,
    DiscreteKnowledgeGradient,
    MCKnowledgeGradient,
    HybridOneShotKnowledgeGradient
)
from botorch.fit import fit_gpytorch_model
from botorch.generation import get_best_candidates, gen_candidates_torch
from botorch.models import SingleTaskGP
from botorch.optim import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from botorch.test_functions import Rosenbrock
from botorch.utils import standardize
from botorch.utils.sampling import manual_seed
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

fun = Rosenbrock(dim=2, negate=True)
fun.bounds[0, :].fill_(-3)
fun.bounds[1, :].fill_(3)

# Defining box optimisation bounds
bounds = torch.stack([-3 * torch.ones(2), 3 * torch.ones(2)])


def eval_objective(x):
    """This is a helper function we use to unnormalize and evaluate a point"""
    return -torch.sum(x**2, dim=1).unsqueeze(-1)# fun(x).unsqueeze(-1)


# Generate XY data
torch.manual_seed(4)
train_X = bounds[0] + (bounds[ 1] - bounds[ 0]) * torch.rand(8, 2)
train_Y = eval_objective(train_X)

# Fit GP model
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

NUM_RESTARTS = 1
RAW_SAMPLES = 100
# Hybrid KG optimisation with Adam's optimiser
NUM_FANTASIES_ONE_SHOT = 20
hybrid_one_shot_kg = HybridOneShotKnowledgeGradient(model=model,
                                             num_fantasies=NUM_FANTASIES_ONE_SHOT)



NUM_DISCRETE_X = 100
discrete_kg = DiscreteKnowledgeGradient(
    model=model,
    bounds=fun.bounds,
    num_discrete_points=NUM_DISCRETE_X,
    X_discretisation=None,
)


# Optimise acquisition functions
with manual_seed(12):
    # Hybrid KG optimisation with Adam's optimiser
    start = time.time()
    xstar_hybrid_kg, _ = optimize_acqf(
        acq_function=hybrid_one_shot_kg,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("hybrid kg done: ", stop - start, "secs")

    start = time.time()
    xstar_discrete_kg, _ = optimize_acqf(
        acq_function=discrete_kg,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("discrete kg done", stop - start, "secs")

# Plotting acquisition functions
x_plot = bounds[ 0] + (bounds[1] - bounds[0]) * torch.rand(5000, 2)
x_plot = x_plot.unsqueeze(dim=-2)

with torch.no_grad():
    discrete_kg_vals = discrete_kg(x_plot)
    discrete_kg_vals = discrete_kg_vals.detach().squeeze().numpy()

plt.scatter(x_plot[:, :, 0], x_plot[:, :, 1], c=discrete_kg_vals)
plt.scatter(
    train_X.numpy()[:, 0], train_X.numpy()[:, 1], color="red", label="sampled points"
)

plt.scatter(
    xstar_hybrid_kg.numpy()[:, 0],
    xstar_hybrid_kg.numpy()[:, 1],
    color="black",
    label="hybrid kg $x^{*}$",
    marker="^",
)

plt.scatter(
    xstar_discrete_kg.numpy()[:, 0],
    xstar_discrete_kg.numpy()[:, 1],
    color="black",
    label="discrete kg $x^{*}$",
)

plt.legend()
plt.colorbar()
plt.show()