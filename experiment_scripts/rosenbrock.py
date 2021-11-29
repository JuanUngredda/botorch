import torch
from botorch.acquisition import qKnowledgeGradient
from botorch.acquisition.analytic import DiscreteKnowledgeGradient
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Rosenbrock
from botorch.utils import standardize
from botorch.utils.sampling import manual_seed
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt

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

def one_shot_knowledge_gradient(model, num_fantasies=100):
    """
    # Defining the qKnowledgeGradient acquisition function (One-Shot KG)
    """

    kg = qKnowledgeGradient(model, num_fantasies=num_fantasies)
    return kg

def discrete_knowledge_gradient(model, bounds, num_discrete_points):
    """
    Computes discrete KG
    """
    kg = DiscreteKnowledgeGradient(model=model,
                                   bounds=bounds,
                                   num_discrete_points=num_discrete_points,
                                   discretisation=None)
    return kg

# Fit GP model
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

# Initialize acquisition functions.
one_shot_kg = one_shot_knowledge_gradient(model=model, num_fantasies=125)
discrete_kg = discrete_knowledge_gradient(model=model, bounds=fun.bounds, num_discrete_points=100)

#Optimise acquisition functions
NUM_RESTARTS = 15
RAW_SAMPLES = 250

with manual_seed(12):
    discrete_kg_xstar, _ = optimize_acqf(
        acq_function=discrete_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    one_shot_kg_xstar, _ = optimize_acqf(
        acq_function=one_shot_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )


# Plotting acquisition functions
plot_x = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(5000, 1, 2)

discrete_kg_vals = discrete_kg(plot_x)
discrete_kg_vals = discrete_kg_vals.detach().squeeze().numpy()

plt.scatter(plot_x[:, :, 0], plot_x[:, :, 1], c=discrete_kg_vals)
plt.scatter(train_X.numpy()[:, 0], train_X.numpy()[:, 1], color="red", label="sampled points")
plt.scatter(discrete_kg_xstar.numpy()[:, 0],
            discrete_kg_xstar.numpy()[:, 1],
            color="black", label="dkg $x^{*}$")
plt.scatter(one_shot_kg_xstar.numpy()[:, 0],
            one_shot_kg_xstar.numpy()[:, 1],
            color="black", marker="x", label="oneshot_kg $x^{*}$")
plt.legend()
plt.colorbar()
plt.show()
