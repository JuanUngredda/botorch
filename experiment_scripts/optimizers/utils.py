import time
from typing import Optional, Callable, Dict, Tuple

import torch
from botorch.acquisition import (
    qKnowledgeGradient,
    HybridKnowledgeGradient,
    DiscreteKnowledgeGradient,
    MCKnowledgeGradient,
)
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _construct_dist
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.model import Model
from botorch.utils import standardize
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


#################################################################
#                                                               #
#                           LHC SAMPLERS                        #
#                                                               #
#################################################################


def lhc(
    n: int,
    dim: Optional[int] = None,
    lb: Optional[Tensor] = None,
    ub: Optional[Tensor] = None,
) -> Tensor:
    """
    Parameters
    ----------
    n: sample size
    dim: optional, dimenions of the cube
    lb: lower bound, Tensor
    ub: upper bound, Tensor
    Returns
    -------
    x: Tensor, shape (n, dim)
    """

    if dim is not None:
        assert (lb is None) and (ub is None), "give dim OR bounds"
        lb = torch.zeros(dim)
        ub = torch.ones(dim)

    else:
        assert (lb is not None) and (ub is not None), "give dim OR bounds"
        lb = lb.squeeze()
        ub = ub.squeeze()
        dim = len(lb)
        assert len(lb) == len(ub), f"bounds are not same shape:{len(lb)}!={len(ub)}"

    x = torch.zeros((n, dim))
    if n > 0:
        for d in range(dim):
            x[:, d] = (torch.randperm(n) + torch.rand(n)) * (1 / n)
            x[:, d] = (ub[d] - lb[d]) * x[:, d] + lb[d]

    return x


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            if name in kw["log_time"].keys():
                kw["log_time"][name].append((te - ts))
            else:
                kw["log_time"][name] = [te - ts]
        return result

    return timed


def KG_wrapper(
    method: str,
    bounds: Optional[Tensor] = None,
    num_fantasies: Optional[int] = None,
    num_discrete_points: Optional[int] = None,
    num_restarts: Optional[int] = None,
    raw_samples: Optional[int] = None,
):
    def acquisition_function(model: method):
        if method == "DISCKG":
            KG_acq_fun = DiscreteKnowledgeGradient(
                model=model,
                bounds=bounds,
                num_discrete_points=num_discrete_points,
                X_discretisation=None,
            )
        elif method == "MCKG":

            KG_acq_fun = MCKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts if num_restarts is not None else 4,
                raw_samples=raw_samples if raw_samples is not None else 80,
            )
        elif method == "HYBRIDKG":
            KG_acq_fun = HybridKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts if num_restarts is not None else 4,
                raw_samples=raw_samples if raw_samples is not None else 80,
            )
        elif method == "ONESHOTKG":
            KG_acq_fun = qKnowledgeGradient(model, num_fantasies=num_fantasies)
        else:
            raise Exception(
                "method does not exist. Specify implemented method: DISCKG (Discrete KG), "
                "MCKG (Monte Carlo KG), HYBRIDKG (Hybrid KG), and ONESHOTKG (One Shot KG)"
            )
        return KG_acq_fun

    return acquisition_function


class ConstrainedPosteriorMean(AnalyticAcquisitionFunction):
    r"""Constrained Posterior Mean (feasibility-weighted).

    Computes the analytic Posterior Mean for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports the case `q=1`. The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.
    """

    def __init__(
        self,
        model: Model,
        objective_index: int,
        maximize: bool = True,
        scalarization=Callable,
    ) -> None:
        r"""Analytic Constrained Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.scalarization = scalarization
        default_value = (None, 0)
        constraints = dict.fromkeys(
            range(model.num_outputs - self.objective_index), default_value
        )
        self._preprocess_constraint_bounds(constraints=constraints)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self._get_posterior(X=X)
        means = posterior.mean.squeeze(dim=-2)  # (b) x m
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        # (b) x 1
        oi = self.objective_index
        mean_obj = means[..., :oi]

        scalarized_objective = self.scalarization(mean_obj)
        mean_constraints = means[..., oi:]
        sigma_constraints = sigmas[..., oi:]

        prob_feas = self._compute_prob_feas(
            X=X.squeeze(dim=-2),
            means=mean_constraints.squeeze(dim=-2),
            sigmas=sigma_constraints.squeeze(dim=-2),
        )

        constrained_posterior_mean = scalarized_objective.mul(prob_feas)
        return constrained_posterior_mean.squeeze(dim=-1)

    def _preprocess_constraint_bounds(
        self, constraints: Dict[int, Tuple[Optional[float], Optional[float]]]
    ) -> None:
        r"""Set up constraint bounds.

        Args:
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
        """
        con_lower, con_lower_inds = [], []
        con_upper, con_upper_inds = [], []
        con_both, con_both_inds = [], []
        con_indices = list(constraints.keys())
        if len(con_indices) == 0:
            raise ValueError("There must be at least one constraint.")
        if self.objective_index in con_indices:
            raise ValueError(
                "Output corresponding to objective should not be a constraint."
            )
        for k in con_indices:
            if constraints[k][0] is not None and constraints[k][1] is not None:
                if constraints[k][1] <= constraints[k][0]:
                    raise ValueError("Upper bound is less than the lower bound.")
                con_both_inds.append(k)
                con_both.append([constraints[k][0], constraints[k][1]])
            elif constraints[k][0] is not None:
                con_lower_inds.append(k)
                con_lower.append(constraints[k][0])
            elif constraints[k][1] is not None:
                con_upper_inds.append(k)
                con_upper.append(constraints[k][1])
        # tensor-based indexing is much faster than list-based advanced indexing
        self.register_buffer("con_lower_inds", torch.tensor(con_lower_inds))
        self.register_buffer("con_upper_inds", torch.tensor(con_upper_inds))
        self.register_buffer("con_both_inds", torch.tensor(con_both_inds))
        # tensor indexing
        self.register_buffer("con_both", torch.tensor(con_both, dtype=torch.float))
        self.register_buffer("con_lower", torch.tensor(con_lower, dtype=torch.float))
        self.register_buffer("con_upper", torch.tensor(con_upper, dtype=torch.float))

    def _compute_prob_feas(self, X: Tensor, means: Tensor, sigmas: Tensor) -> Tensor:
        r"""Compute feasibility probability for each batch of X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            means: A `(b) x m`-dim Tensor of means.
            sigmas: A `(b) x m`-dim Tensor of standard deviations.
        Returns:
            A `(b) x 1`-dim tensor of feasibility probabilities

        Note: This function does case-work for upper bound, lower bound, and both-sided
        bounds. Another way to do it would be to use 'inf' and -'inf' for the
        one-sided bounds and use the logic for the both-sided case. But this
        causes an issue with autograd since we get 0 * inf.
        TODO: Investigate further.
        """
        output_shape = X.shape[:-2] + torch.Size([1])
        prob_feas = torch.ones(output_shape, device=X.device, dtype=X.dtype)

        if len(self.con_lower_inds) > 0:
            self.con_lower_inds = self.con_lower_inds.to(device=X.device)
            normal_lower = _construct_dist(means, sigmas, self.con_lower_inds)
            prob_l = 1 - normal_lower.cdf(self.con_lower)
            prob_feas = prob_feas.mul(torch.prod(prob_l, dim=-1, keepdim=True))
        if len(self.con_upper_inds) > 0:
            self.con_upper_inds = self.con_upper_inds.to(device=X.device)
            normal_upper = _construct_dist(means, sigmas, self.con_upper_inds)
            prob_u = normal_upper.cdf(self.con_upper)
            prob_feas = prob_feas.mul(torch.prod(prob_u, dim=-1, keepdim=True))
        if len(self.con_both_inds) > 0:
            self.con_both_inds = self.con_both_inds.to(device=X.device)
            normal_both = _construct_dist(means, sigmas, self.con_both_inds)
            prob_u = normal_both.cdf(self.con_both[:, 1])
            prob_l = normal_both.cdf(self.con_both[:, 0])
            prob_feas = prob_feas.mul(torch.prod(prob_u - prob_l, dim=-1, keepdim=True))
        return prob_feas


def ParetoFrontApproximation(
    model: Model,
    input_dim: int,
    scalatization_fun: Callable,
    bounds: Tensor,
    y_train: Tensor,
    weights: Tensor,
    optional: Optional[dict[str, int]] = None,
) -> Tensor:
    y_train_standarized = standardize(y_train)
    X_pareto_solutions = []
    # X_random_start = []
    for w in weights:
        # normalizes scalariztion between [0,1] given the training data.
        scalarization = scalatization_fun(weights=w, Y=y_train_standarized)

        constrained_model = ConstrainedPosteriorMean(
            model=model,
            objective_index=y_train.shape[-1],
            scalarization=scalarization,
        )

        domain_offset = bounds[0]
        domain_range = bounds[1] - bounds[0]
        X_unit_cube_samples = torch.rand((optional["RAW_SAMPLES"], 1, 1, input_dim))
        X_initial_conditions_raw = X_unit_cube_samples * domain_range + domain_offset

        mu_val_initial_conditions_raw = constrained_model.forward(
            X_initial_conditions_raw
        )

        best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
            : optional["NUM_RESTARTS"]
        ]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]
        top_x_initial_means, value_initial_means = gen_candidates_scipy(
            initial_conditions=X_initial_conditions,
            acquisition_function=constrained_model,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
        )
        # subopt_x = X_initial_conditions[torch.argmax(value_initial_means), ...]
        top_x = top_x_initial_means[torch.argmax(value_initial_means), ...]
        X_pareto_solutions.append(top_x)
        # X_random_start.append(subopt_x)

    X_pareto_solutions = torch.vstack(X_pareto_solutions)
    # X_random_start = torch.vstack(X_random_start)

    # plot_X = torch.rand((1000,3))
    # posterior = self.model.posterior(plot_X)
    # mean = posterior.mean.detach().numpy()
    # is_feas = (mean[:,2] <= 0)
    # print("mean", mean.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(mean[is_feas,0], mean[is_feas,1], c=mean[is_feas,2])
    #
    # Y_pareto_posterior = self.model.posterior(X_pareto_solutions)
    # Y_pareto_mean = Y_pareto_posterior.mean.detach().numpy()
    # print(Y_pareto_mean.shape)
    # plt.scatter(Y_pareto_mean[...,0], Y_pareto_mean[...,1], color="red")
    #
    # Y_pareto_posterior = self.model.posterior(X_random_start)
    # Y_pareto_mean = Y_pareto_posterior.mean.detach().numpy()
    # print(Y_pareto_mean.shape)
    # plt.scatter(Y_pareto_mean[...,0], Y_pareto_mean[...,1], color="red", marker="x")
    #
    # plt.show()
    # raise
    # raise
    return X_pareto_solutions


def _compute_expected_utility(
    objective: Callable,
    scalatization_fun: Callable,
    y_values: Tensor,
    c_values: Tensor,
    weights: Tensor,
) -> Tensor:

    objective_func_front = objective.gen_pareto_front(n=20)
    utility = torch.zeros((weights.shape[0], y_values.shape[0]))

    for idx, w in enumerate(weights):
        scalarization = scalatization_fun(weights=w, Y=objective_func_front)
        utility_values = scalarization(y_values).squeeze()
        constraint_binary_values = (c_values <= 0).type_as(utility_values).squeeze()

        utility[idx, :] = utility_values * constraint_binary_values

    expected_utility = utility.mean()
    return expected_utility
