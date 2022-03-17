import matplotlib.pyplot as plt
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP
from botorch.test_functions import Rosenbrock
from botorch.utils.transforms import unnormalize

fun = Rosenbrock(dim=1, negate=True)
fun.bounds[0].fill_(-5)
fun.bounds[1].fill_(10)

# Defining box optimisation bounds
bounds = torch.stack([-5 * torch.ones(2), 10 * torch.ones(2)])


def eval_objective(x):
    """This is a helper function we use to unnormalize and evaluate a point"""
    x_unnormed = unnormalize(x, torch.Tensor([-2.7, 7.5])).squeeze()
    return (torch.sin(x_unnormed) + torch.sin((10.0 / 3.0) * x_unnormed)).unsqueeze(-1)


n_init_design = 4
# Generate XY data
torch.manual_seed(1)
train_X = torch.linspace(0.1, 0.90, n_init_design).unsqueeze(-1)
train_Y = eval_objective(train_X)

# Fit GP model
# print(train_X.shape)
# print(train_Y.shape)
NOISE_VAR = torch.Tensor([1e-4])

model = FixedNoiseGP(train_X, train_Y, NOISE_VAR.expand_as(train_Y))
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

X_plot = torch.linspace(0, 1, 100).unsqueeze(dim=-1)
true_fvals = eval_objective(X_plot).detach().numpy()
# print(X_plot.shape)
posterior = model.posterior(X_plot)
mean = posterior.mean.detach().numpy().reshape(-1)
variance = posterior.variance.detach().numpy().reshape(-1)

import numpy as np

X_plot = X_plot.detach().numpy().reshape(-1)
plt.plot(X_plot, mean + 1.95 * np.sqrt(variance), color="grey", alpha=0.6)
plt.plot(X_plot, mean - 1.95 * np.sqrt(variance), color="grey", alpha=0.6)
plt.fill_between(x=X_plot, y1=mean - 1.95 * np.sqrt(variance), y2=mean + 1.95 * np.sqrt(variance),
                 color="salmon", alpha=0.2, label="GP 95% CI")
plt.plot(X_plot, mean, color="black", linestyle='--', label="GP mean")
plt.plot(X_plot, true_fvals, color="green", label="black-box function")
plt.scatter(train_X, train_Y, color="magenta", edgecolors="black", s=60, label="initial design")
plt.scatter(X_plot[np.argmax(true_fvals)], np.max(true_fvals), marker=(5, 1), edgecolors="black", color="yellow", s=300,
            label="true best design")
plt.legend(prop={'size': 10})
plt.xlim(0, 1)
plt.xlabel("$\mathbb{X}$", size=24)
plt.savefig("/home/juan/Documents/repos_data/PhD_Thesis/pictures/GP_regression_init_design.pdf", bbox_inches='tight')
plt.show()

for i in range(0, 50):
    from botorch.acquisition import ExpectedImprovement

    best_value = train_Y.max()
    print("best val", best_value)
    EI = ExpectedImprovement(model=model, best_f=best_value, maximize=True)

    X_plot = torch.sort(torch.rand((1000, 1, 1)), dim=0).values
    EI_vals = EI.forward(X_plot)
    x_best = X_plot[torch.argmax(EI_vals.squeeze())]

    acq_vals = (EI_vals.detach().numpy() - np.min(EI_vals.detach().numpy()))/(np.max(EI_vals.detach().numpy())-np.min(EI_vals.detach().numpy()))

    plt.plot(X_plot.squeeze().detach().numpy(), acq_vals)

    plt.vlines(x=X_plot.squeeze().detach().numpy()[np.argmax(acq_vals)], ymin=-0.01, ymax=np.max(acq_vals),
               color="red", linestyles="--", label="$$")
    plt.scatter(X_plot.squeeze().detach().numpy()[np.argmax(acq_vals)], np.max(acq_vals), marker=(5, 2), color="red",
                s=100)
    plt.xlabel("$\mathbb{X}$", size=24)
    plt.ylabel(r"$\alpha$(x)", size=24)
    plt.xlim((0, 1))
    plt.ylim((-0.01, np.max(acq_vals) + 0.02))
    plt.savefig("/home/juan/Documents/repos_data/PhD_Thesis/pictures/acq_it_{}.pdf".format(i), bbox_inches='tight')
    plt.show()

    train_X = torch.cat([train_X, x_best])
    new_y = torch.atleast_2d(eval_objective(x_best))
    train_Y = torch.cat([train_Y, new_y])

    model = FixedNoiseGP(train_X, train_Y, NOISE_VAR.expand_as(train_Y))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    X_plot = torch.linspace(0, 1, 100).unsqueeze(dim=-1)
    true_fvals = eval_objective(X_plot).detach().numpy()
    # print(X_plot.shape)
    posterior = model.posterior(X_plot)
    mean = posterior.mean.detach().numpy().reshape(-1)
    variance = posterior.variance.detach().numpy().reshape(-1)
    X_plot = X_plot.detach().numpy().reshape(-1)
    plt.plot(X_plot, mean + 1.95 * np.sqrt(variance), color="grey", alpha=0.6)
    plt.plot(X_plot, mean - 1.95 * np.sqrt(variance), color="grey", alpha=0.6)
    plt.fill_between(x=X_plot, y1=mean - 1.95 * np.sqrt(variance), y2=mean + 1.95 * np.sqrt(variance),
                     color="salmon", alpha=0.2, label="GP 95% CI")
    plt.plot(X_plot, mean, color="black", linestyle='--', label="GP mean")
    plt.plot(X_plot, true_fvals, color="green", label="black-box function")
    plt.scatter(train_X[:n_init_design], train_Y[:n_init_design], color="magenta", edgecolors="black", s=60,
                label="initial design")
    plt.scatter(train_X[n_init_design:], train_Y[n_init_design:], color="red", edgecolors="black", s=60,
                label="sampled designs")
    plt.scatter(X_plot[np.argmax(true_fvals)], np.max(true_fvals), marker=(5, 1), edgecolors="black", color="yellow",
                s=300, label="true best design")
    plt.legend(prop={'size': 10})
    plt.xlim(0, 1)
    plt.xlabel("$\mathbb{X}$", size=24)
    plt.savefig("/home/juan/Documents/repos_data/PhD_Thesis/pictures/GP_regression_it_{}.pdf".format(i), bbox_inches='tight')
    plt.show()
    # raise
# for
# print(mean, variance)
raise
# acquisition function and optimisation parameters
NUM_RESTARTS = 5
RAW_SAMPLES = 80

# Initialize acquisition functions.
NUM_FANTASIES_ONE_SHOT = 125
one_shot_kg = qKnowledgeGradient(model, num_fantasies=NUM_FANTASIES_ONE_SHOT)

NUM_DISCRETE_X = 100
discrete_kg = DiscreteKnowledgeGradient(
    model=model,
    bounds=fun.bounds,
    num_discrete_points=NUM_DISCRETE_X,
    X_discretisation=None,
)

NUM_FANTASIES_CONTINUOUS_KG = 5
continuous_kg = MCKnowledgeGradient(
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
    xstar_hybrid_kg, _ = optimize_acqf(
        acq_function=hybrid_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("hybrid kg done: ", stop - start, "secs")

    start = time.time()
    xstar_discrete_kg, _ = optimize_acqf(
        acq_function=discrete_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("discrete kg done", stop - start, "secs")

    start = time.time()
    xstar_one_shot_kg, _ = optimize_acqf(
        acq_function=one_shot_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    stop = time.time()
    print("one-shot kg done", stop - start, "secs")

    # Hybrid KG optimisation with deterministic optimiser
    start = time.time()
    initial_conditions = gen_batch_initial_conditions(
        acq_function=continuous_kg,
        bounds=bounds.T,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    batch_candidates, batch_acq_values = gen_candidates_torch(
        initial_conditions=initial_conditions,
        acquisition_function=continuous_kg,
        lower_bounds=bounds.T[0],
        upper_bounds=bounds.T[1],
        optimizer=torch.optim.Adam,
        verbose=True,
        options={"maxiter": 100},
    )
    xstar_continuous_kg = get_best_candidates(
        batch_candidates=batch_candidates, batch_values=batch_acq_values
    ).detach()

    stop = time.time()
    print("continuous kg done: ", stop - start, "secs")

# Plotting acquisition functions
x_plot = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(5000, 1, 2)

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
plt.scatter(
    xstar_one_shot_kg.numpy()[:, 0],
    xstar_one_shot_kg.numpy()[:, 1],
    color="black",
    marker="x",
    label="oneshot_kg $x^{*}$",
)
plt.scatter(
    xstar_continuous_kg.numpy()[:, 0],
    xstar_continuous_kg.numpy()[:, 1],
    color="black",
    marker="s",
    label="continuous_kg $x^{*}$",
)
plt.legend()
plt.colorbar()
plt.show()
