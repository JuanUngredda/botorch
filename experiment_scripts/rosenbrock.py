import time

import matplotlib.pyplot as plt
import torch
from botorch.acquisition import (
    qKnowledgeGradient,
    HybridKnowledgeGradient,
    DiscreteKnowledgeGradient,
    ContinuousKnowledgeGradient,
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
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)

# Defining box optimisation bounds
bounds = torch.stack([torch.zeros(2), torch.ones(2)])
bounds[:, 0].fill_(-5)
bounds[:, 1].fill_(10)


def eval_objective(x):
    """This is a helper function we use to unnormalize and evaluate a point"""
    return fun(unnormalize(x, fun.bounds)).unsqueeze(-1)


# Generate XY data
torch.manual_seed(1)
train_X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(30, 2)
train_Y = standardize(eval_objective(train_X))

# Fit GP model
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

# acquisition function and optimisation parameters
NUM_RESTARTS = 5
RAW_SAMPLES = 50

# Initialize acquisition functions.
NUM_FANTASIES_ONE_SHOT = 125
one_shot_kg = qKnowledgeGradient(model, num_fantasies=NUM_FANTASIES_ONE_SHOT)

NUM_DISCRETE_X = 100
discrete_kg = DiscreteKnowledgeGradient(
    model=model,
    bounds=fun.bounds,
    num_discrete_points=NUM_DISCRETE_X,
    discretisation=None,
)

NUM_FANTASIES_CONTINUOUS_KG = 5
continous_kg = ContinuousKnowledgeGradient(
    model,
    bounds=fun.bounds,
    num_fantasies=NUM_FANTASIES_CONTINUOUS_KG,
    num_restarts=1,
    raw_samples=20,
)


NUM_FANTASIES_HYBRID_KG = 5
hybrid_kg = HybridKnowledgeGradient(
    model=model,
    bounds=fun.bounds,
    num_fantasies=NUM_FANTASIES_HYBRID_KG,
    num_restarts=1,
    raw_samples=20,
)

# Optimise acquisition functions
with manual_seed(12):

    # Hybrid KG optimisation with Adam's optimiser
    start = time.time()
    hybrid_kg_xstar, _ = optimize_acqf(
        acq_function=hybrid_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("hybrid kg done: ", stop - start, "secs")

    # Hybrid KG optimisation with deterministic optimiser
    start = time.time()
    initial_conditions = gen_batch_initial_conditions(
        acq_function=continous_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    batch_candidates, batch_acq_values = gen_candidates_torch(
        initial_conditions=initial_conditions,
        acquisition_function=continous_kg,
        lower_bounds=bounds.T[0],
        upper_bounds=bounds.T[1],
        optimizer=torch.optim.Adam,
        verbose=True,
        options={"maxiter": 100},
    )
    continuous_kg_xstar = get_best_candidates(
        batch_candidates=batch_candidates, batch_values=batch_acq_values
    ).detach()

    # continuous_kg_xstar, _ = optimize_acqf(
    #     acq_function=continous_kg, bounds=bounds.T, q=1, num_restarts=5, raw_samples=50
    # )
    stop = time.time()
    # print("continuous kg done: ", stop - start, "secs")

    start = time.time()
    discrete_kg_xstar, _ = optimize_acqf(
        acq_function=discrete_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("discrete kg done", stop - start, "secs")

    start = time.time()
    one_shot_kg_xstar, _ = optimize_acqf(
        acq_function=one_shot_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("one-shot kg done", stop - start, "secs")

# Plotting acquisition functions
x_plot = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(5000, 1, 2)

discrete_kg_vals = discrete_kg(x_plot)
discrete_kg_vals = discrete_kg_vals.detach().squeeze().numpy()

plt.scatter(x_plot[:, :, 0], x_plot[:, :, 1], c=discrete_kg_vals)
plt.scatter(
    train_X.numpy()[:, 0], train_X.numpy()[:, 1], color="red", label="sampled points"
)

plt.scatter(
    hybrid_kg_xstar.numpy()[:, 0],
    hybrid_kg_xstar.numpy()[:, 1],
    color="black",
    label="hybrid kg $x^{*}$",
    marker="^",
)

plt.scatter(
    discrete_kg_xstar.numpy()[:, 0],
    discrete_kg_xstar.numpy()[:, 1],
    color="black",
    label="discrete kg $x^{*}$",
)
plt.scatter(
    one_shot_kg_xstar.numpy()[:, 0],
    one_shot_kg_xstar.numpy()[:, 1],
    color="black",
    marker="x",
    label="oneshot_kg $x^{*}$",
)
plt.scatter(
    continuous_kg_xstar.numpy()[:, 0],
    continuous_kg_xstar.numpy()[:, 1],
    color="black",
    marker="s",
    label="continuous_kg $x^{*}$",
)
plt.legend()
plt.colorbar()
plt.show()
