import os

import gpytorch
import torch
from botorch.acquisition import qKnowledgeGradient
from botorch.acquisition.analytic import qDiscreteKnowledgeGradient
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Rosenbrock
from botorch.utils import standardize
from botorch.utils.sampling import manual_seed
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood


# Defining box optimisation bounds
bounds = torch.stack([torch.zeros(1), torch.ones(1)])

# Generate XY data
train_X = -2.7 + (7.5 - -2.7) * torch.rand(30, 1)
fun = Rosenbrock(dim=2, negate=True)
fun.bounds[0, :].fill_(-2.7)
fun.bounds[1, :].fill_(7.5)

# objective function
def eval_objective(x):
    """This is a helper function we use to unnormalize and evaluate a point"""
    return torch.sin(x) + torch.sin(
        (10.0 / 3.0) * x
    )  # fun(unnormalize(x, fun.bounds)).unsqueeze(-1)


train_Y = standardize(eval_objective(train_X))


# Fit GP model
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)


from botorch.optim.initializers import gen_batch_initial_conditions

ics = -2.7 + (7.5 - -2.7) * torch.rand(1000, 1)


print(ics.shape)
posterior = model.posterior(ics, observation_noise=False)
mean = posterior.mean
# deal with batch evaluation and broadcasting
print(mean.shape)
var = posterior.variance.clamp_min(1e-9)
model_noise_var = model.likelihood.noise
print("model_noise_var", model_noise_var)

cov_f = posterior.mvn.covariance_matrix

variance = cov_f.diag()
import matplotlib.pyplot as plt
import numpy as np

m = mean.detach().squeeze().numpy()
v = variance.detach().squeeze().numpy()
plt.scatter(ics.detach().squeeze().numpy(), m + 1.96 * np.sqrt(v), s=1)
plt.scatter(ics.detach().squeeze().numpy(), m - 1.96 * np.sqrt(v), s=1)
plt.scatter(ics.detach().squeeze().numpy(), m, s=1)
plt.scatter(train_X.squeeze(), train_Y.squeeze(), color="black")
plt.show()

ics_150 = -2.7 + (7.5 - -2.7) * torch.rand(150, 1)
ics_overall = torch.cat([ics, ics_150])
print(ics_overall.shape)

posterior = model.posterior(ics_overall, observation_noise=False)

cov_submatrix_f = posterior.mvn.covariance_matrix
print(cov_submatrix_f.shape)
cov_matrix = cov_submatrix_f[:1000, :1000]
var_matrix = cov_matrix.diag()

m = mean.detach().squeeze().numpy()
v = var_matrix.detach().squeeze().numpy()
plt.scatter(ics.detach().squeeze().numpy(), m + 1.96 * np.sqrt(v), s=1)
plt.scatter(ics.detach().squeeze().numpy(), m - 1.96 * np.sqrt(v), s=1)
plt.scatter(ics.detach().squeeze().numpy(), m, s=1)
plt.scatter(train_X.squeeze(), train_Y.squeeze(), color="black")
plt.show()

raise

# Defining the qKnowledgeGradient acquisition function (One-Shot KG)
NUM_FANTASIES = 100
qKG = qKnowledgeGradient(model, num_fantasies=NUM_FANTASIES)

# Optimize acquisition function
NUM_RESTARTS = 5000
RAW_SAMPLES = 10000


dKG = qDiscreteKnowledgeGradient(model=model, bounds=fun.bounds, num_points=100)

from botorch.optim.initializers import gen_batch_initial_conditions

ics = gen_batch_initial_conditions(
    acq_function=dKG,
    bounds=fun.bounds,
    q=1,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
)

qKG_val = dKG(ics)

print(qKG_val)
raise
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions

ics = gen_one_shot_kg_initial_conditions(
    acq_function=qKG,
    bounds=fun.bounds,
    q=1,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
)

print(ics.shape)
acq = qKG.forward(ics)

X_actual = qKG.extract_candidates(X_full=ics)
X_actual = X_actual.detach().squeeze().numpy()
acq = acq.detach().squeeze().numpy()
print(acq)

import matplotlib.pyplot as plt

plt.scatter(X_actual[:, 0], X_actual[:, 1], c=acq)
plt.show()
raise

with manual_seed(1234):
    candidates, acq_value = optimize_acqf(
        acq_function=qKG,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

print("candidates: ", candidates, "acq_value ", acq_value)
