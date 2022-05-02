import time
from typing import Optional, Callable, Dict, Tuple

import torch
from torch import Tensor
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _construct_dist
from botorch.acquisition.multi_objective.multi_attribute_constrained_kg import MultiAttributeConstrainedKG, MultiAttributePenalizedKG
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.model import Model
from botorch.utils import standardize
from botorch.utils.transforms import t_batch_mode_transform
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.scalarization import (
    get_chebyshev_scalarization,
    get_linear_scalarization,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import qMultiObjectiveMaxValueEntropy
from botorch import settings

dtype = torch.double


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


def test_function_handler(test_fun_str: str,
                          test_fun_dict: dict,
                          input_dim: int,
                          output_dim: int):
    if test_fun_str == "C2DTLZ2":
        synthetic_fun = test_fun_dict[test_fun_str](dim=input_dim,
                                                    num_objectives=output_dim,
                                                    negate=True)
    else:

        synthetic_fun = test_fun_dict[test_fun_str](negate=True)

    return synthetic_fun


def get_constrained_mc_objective(train_obj, train_con, scalarization, num_constraints):
    """Initialize a ConstrainedMCObjective for qParEGO"""
    n_obj = train_obj.shape[-1]

    # assume first outcomes of the model are the objectives, the rest constraints
    def objective(Z):
        return scalarization(Z[..., :n_obj])

    constrained_obj = ConstrainedMCObjective(
        objective=objective,
        constraints=[lambda Z: Z[..., -(i+1)] for i in range(num_constraints)],  # index the constraints
    )
    return constrained_obj


def mo_acq_wrapper(
        method: str,
        test_fun: Callable,
        num_objectives: int,
        utility_model_name=str,
        bounds: Optional[Tensor] = None,
        MC_size: Optional[int] = None,
        num_scalarizations: Optional[int] = None,
        num_discrete_points: Optional[int] = None,
        num_restarts: Optional[int] = None,
        raw_samples: Optional[int] = None,
):
    if utility_model_name == "Tche":
        utility_model = get_chebyshev_scalarization

    elif utility_model_name == "Lin":
        utility_model = get_linear_scalarization

    def acquisition_function(model: method,
                             train_x: Tensor,
                             train_obj: Tensor,
                             train_con: Tensor,
                             fixed_scalarizations: Tensor,
                             current_global_optimiser: Tensor,
                             X_pending: Optional[Tensor] = None):

        if method == "macKG":
            acq_fun = MultiAttributeConstrainedKG(
                model=model,
                bounds=bounds,
                X_discretisation_size=num_discrete_points,
                utility_model=utility_model,
                num_objectives=num_objectives,
                num_fantasies_constraints=MC_size,
                fixed_scalarizations=fixed_scalarizations,
                num_scalarisations=num_scalarizations,
                current_global_optimiser=current_global_optimiser,
                X_pending=X_pending)

        elif method == "pen-maKG":
            acq_fun = MultiAttributePenalizedKG(
                model=model,
                bounds=bounds,
                X_discretisation_size=num_discrete_points,
                utility_model=utility_model,
                num_objectives=num_objectives,
                num_fantasies_constraints=MC_size,
                fixed_scalarizations=fixed_scalarizations,
                num_scalarisations=num_scalarizations,
                current_global_optimiser=current_global_optimiser,
                X_pending=X_pending)

        elif method == "EHI":
            with torch.no_grad():
                model_obj = model.subset_output(idcs=range(num_objectives))
                qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_size)  # 128 samples.
                pred = model_obj.posterior(train_x).mean

                partitioning = FastNondominatedPartitioning(
                    ref_point=test_fun.ref_point,
                    Y=pred,
                )
                acq_fun = qExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=test_fun.ref_point,
                    partitioning=partitioning,
                    # define an objective that specifies which outcomes are the objectives
                    objective=IdentityMCMultiOutputObjective(outcomes=range(test_fun.num_objectives)),
                    sampler=qehvi_sampler,
                )

        elif method == "ParEGO":
            model_obj = model.subset_output(idcs=range(num_objectives))
            with torch.no_grad():
                pred = model_obj.posterior(train_x).mean
            qparego_sampler = SobolQMCNormalSampler(num_samples=MC_size)  # 128 samples

            weights = sample_simplex(d=test_fun.num_objectives, n=1).squeeze()

            objective = GenericMCObjective(utility_model(weights=weights, Y=pred))

            acq_fun = qExpectedImprovement(
                model=model_obj,
                objective=objective,
                best_f=objective(train_obj).max(),
                sampler=qparego_sampler,
            )


        elif method == "cEHI":
            with torch.no_grad():
                model_obj = model.subset_output(idcs=range(num_objectives))
                num_constraints = test_fun.num_constraints
                qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_size)  # 128 samples.
                pred = model_obj.posterior(train_x).mean
                partitioning = FastNondominatedPartitioning(
                    ref_point=test_fun.ref_point,
                    Y=pred,
                )

                acq_fun = qExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=test_fun.ref_point,
                    partitioning=partitioning,
                    sampler=qehvi_sampler,
                    # define an objective that specifies which outcomes are the objectives
                    objective=IdentityMCMultiOutputObjective(outcomes=range(test_fun.num_objectives)),
                    # specify that the constraint is on the last outcome
                    constraints=[lambda Z: Z[..., -(i+1)] for i in range(num_constraints)],

                )
        elif method == "cParEGO":
            qparego_sampler = SobolQMCNormalSampler(num_samples=MC_size)  # 128 samples
            num_constraints = test_fun.num_constraints
            weights = sample_simplex(d=test_fun.num_objectives, n=1).squeeze()
            # construct augmented Chebyshev scalarization
            scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)
            # initialize ConstrainedMCObjective
            constrained_objective = get_constrained_mc_objective(
                train_obj=train_obj,
                train_con=train_con,
                scalarization=scalarization,
                num_constraints=num_constraints
            )
            train_y = torch.cat([train_obj, train_con], dim=-1)
            acq_fun = qExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=constrained_objective,
                best_f=constrained_objective(train_y).max(),
                sampler=qparego_sampler,
            )
        else:
            raise Exception(
                "method does not exist. Specify implemented method"
            )
        return acq_fun

    return acquisition_function


class ConstrainedPosteriorMean_individual(AnalyticAcquisitionFunction):
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
            num_constraints: int,
            num_objectives: int
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
        self.objective_index = objective_index
        self.num_objectives = num_objectives
        self.constraints_index = num_constraints

        self.model_obj = self.model.subset_output(idcs=range(self.objective_index, self.objective_index + 1))
        self.model_cs = self.model.subset_output(idcs=range(self.num_objectives, model.num_outputs))
        default_value = (None, 0)
        constraints = dict.fromkeys(
            range(model.num_outputs - self.num_objectives), default_value
        )
        print("constraints", constraints)
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
        X = X.to(dtype=torch.double)
        posterior_obj = self.model_obj.posterior(X=X)
        mean_obj = posterior_obj.mean.squeeze(dim=-2)  # (b) x m

        posterior_cs = self.model_cs.posterior(X=X)
        mean_constraints = posterior_cs.mean.squeeze(dim=-2)  # (b) x m
        sigma_constraints = posterior_cs.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        prob_feas = self._compute_prob_feas(
            X=X.squeeze(dim=-2),
            means=mean_constraints.squeeze(dim=-2),
            sigmas=sigma_constraints.squeeze(dim=-2),
        ).double()
        constrained_posterior_mean = mean_obj.squeeze() * prob_feas.squeeze()

        return constrained_posterior_mean.squeeze(dim=-1).double()

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

        # if self.num_objectives in con_indices:
        #     raise ValueError(
        #         "Output corresponding to objective should not be a constraint."
        #     )
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


class ConstrainedPosteriorMean_individual_threshold(AnalyticAcquisitionFunction):
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
            num_constraints: int,
            num_objectives: int
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
        self.objective_index = objective_index
        self.num_objectives = num_objectives
        self.constraints_index = num_constraints

        self.model_obj = self.model.subset_output(idcs=range(self.objective_index, self.objective_index + 1))
        self.model_cs = self.model.subset_output(idcs=range(self.num_objectives, model.num_outputs))
        default_value = (None, 0)
        constraints = dict.fromkeys(
            range(model.num_outputs - self.num_objectives), default_value
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
        X = X.to(dtype=torch.double)
        posterior_obj = self.model_obj.posterior(X=X)
        mean_obj = posterior_obj.mean.squeeze(dim=-2)  # (b) x m

        posterior_cs = self.model_cs.posterior(X=X)
        mean_constraints = posterior_cs.mean.squeeze(dim=-2)  # (b) x m
        sigma_constraints = posterior_cs.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        prob_feas = self._compute_prob_feas(
            X=X.squeeze(dim=-2),
            means=mean_constraints.squeeze(dim=-2),
            sigmas=sigma_constraints.squeeze(dim=-2),
        ).double()

        # indicator = (prob_feas.squeeze() > (0.49)** self.constraints_index)*1.0

        constrained_posterior_mean = mean_obj.squeeze() * prob_feas.squeeze()  # indicator.squeeze()

        return constrained_posterior_mean.squeeze(dim=-1).double()

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
        #
        # if self.num_objectives in con_indices:
        #     raise ValueError(
        #         "Output corresponding to objective should not be a constraint."
        #     )
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
        X = X.to(dtype=torch.double)
        posterior = self.model.posterior(X=X)
        means = posterior.mean.squeeze(dim=-2)  # (b) x m
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        # (b) x 1
        oi = self.objective_index
        mean_obj = means[..., :oi]
        scalarized_objective = self.scalarization(Y=mean_obj)

        mean_constraints = means[..., oi:]
        sigma_constraints = sigmas[..., oi:]

        prob_feas = self._compute_prob_feas(
            X=X.squeeze(dim=-2),
            means=mean_constraints.squeeze(dim=-2),
            sigmas=sigma_constraints.squeeze(dim=-2),
        ).double()

        constrained_posterior_mean = scalarized_objective.squeeze() * prob_feas.squeeze()

        # import matplotlib.pyplot as plt
        # val =constrained_posterior_mean.squeeze(dim=-1).detach().numpy()
        # plt.scatter(mean_obj.detach().numpy()[..., 0], mean_obj.detach().numpy()[..., 1], c=val)
        # plt.show()
        # raise

        return constrained_posterior_mean.squeeze(dim=-1).double()

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


def ParetoFrontApproximation_xstar(
        model: Model,
        input_dim: int,
        objective_dim: int,
        scalatization_fun: Callable,
        bounds: Tensor,
        y_train: Tensor,
        x_train: Tensor,
        c_train: Tensor,
        weights: Tensor,
        num_objectives: int,
        num_constraints: int,
        optional: Optional[dict[str, int]] = None,
) -> tuple[Tensor, Tensor]:
    X_pareto_solutions = []
    X_pmean = []

    for idx, w in enumerate(weights):
        constrained_model = ConstrainedPosteriorMean_individual_threshold(
            model=model,
            objective_index=idx,
            num_objectives=num_objectives,
            num_constraints=num_constraints
        )

        X_initial_conditions_raw = torch.rand((1000, 1, 1, input_dim))

        with torch.no_grad():
            mu_val_initial_conditions_raw = constrained_model.forward(
                X_initial_conditions_raw
            )

        best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                         : 1
                         ]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :].double()
        # print("exited")
        # print("entered opt")
        top_x_initial_means, value_initial_means = gen_candidates_scipy(
            initial_conditions=X_initial_conditions,
            acquisition_function=constrained_model,
            lower_bounds=torch.zeros(input_dim),
            upper_bounds=torch.ones(input_dim))

        top_x = top_x_initial_means[torch.argmax(value_initial_means), ...]
        X_pareto_solutions.append(top_x)
        X_pmean.append(torch.max(value_initial_means))

    X_pareto_solutions = torch.vstack(X_pareto_solutions)
    X_pmean = torch.vstack(X_pmean)

    #########################################################
    # plot_X = torch.rand((1000,3))
    #
    # from botorch.fit import fit_gpytorch_model
    # from botorch.models import SingleTaskGP
    # from botorch.models.model_list_gp_regression import ModelListGP
    # from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    #
    # Y_train_standarized = standardize(y_train)
    # train_joint_YC = torch.cat([Y_train_standarized, c_train], dim=-1)
    #
    # models = []
    # for i in range(train_joint_YC.shape[-1]):
    #     models.append(
    #         SingleTaskGP(x_train, train_joint_YC[..., i: i + 1])
    #     )
    # model = ModelListGP(*models)
    # mll = SumMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_model(mll)
    #
    # posterior = model.posterior(plot_X)
    # mean = posterior.mean.detach().numpy()
    # is_feas = (mean[:,2] <= 0)
    # print("weights", weights)
    # mu_val_initial_conditions_raw = constrained_model.forward(plot_X.unsqueeze(dim=-2)).detach().numpy()
    #
    # import matplotlib.pyplot as plt
    # plt.scatter(mean[is_feas,0], mean[is_feas,1], c=mu_val_initial_conditions_raw.squeeze()[is_feas])
    #
    # Y_pareto_posterior = model.posterior(X_pareto_solutions)
    # Y_pareto_mean = Y_pareto_posterior.mean.detach().numpy()
    # print(Y_pareto_mean.shape)
    # plt.scatter(Y_pareto_mean[...,0], Y_pareto_mean[...,1], color="red")
    #
    # plt.show()
    # raise

    return X_pareto_solutions, X_pmean


def ParetoFrontApproximation(
        model: Model,
        input_dim: int,
        objective_dim: int,
        scalatization_fun: Callable,
        bounds: Tensor,
        y_train: Tensor,
        x_train: Tensor,
        c_train: Tensor,
        weights: Tensor,
        num_objectives: int,
        num_constraints: int,
        optional: Optional[dict[str, int]] = None,
) -> tuple[Tensor, Tensor]:
    X_pareto_solutions = []
    X_pmean = []

    for idx, w in enumerate(weights):
        print("idx", idx)
        print("num_obj", num_objectives)
        print("num_const", num_constraints)
        print("model outpus", model.num_outputs)
        constrained_model = ConstrainedPosteriorMean_individual(
            model=model,
            objective_index=idx,
            num_objectives=num_objectives,
            num_constraints=num_constraints
        )

        X_initial_conditions_raw = torch.rand((optional["RAW_SAMPLES"], 1, 1, input_dim))

        mu_val_initial_conditions_raw = constrained_model.forward(
            X_initial_conditions_raw
        )

        best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                         : optional["NUM_RESTARTS"]
                         ]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :].double()

        top_x_initial_means, value_initial_means = gen_candidates_scipy(
            initial_conditions=X_initial_conditions,
            acquisition_function=constrained_model,
            lower_bounds=torch.zeros(input_dim),
            upper_bounds=torch.ones(input_dim))

        top_x = top_x_initial_means[torch.argmax(value_initial_means), ...]
        X_pareto_solutions.append(top_x)
        X_pmean.append(torch.max(value_initial_means))

    X_pareto_solutions = torch.vstack(X_pareto_solutions)
    X_pmean = torch.vstack(X_pmean)

    #########################################################
    # plot_X = torch.rand((1000,3))
    #
    # from botorch.fit import fit_gpytorch_model
    # from botorch.models import SingleTaskGP
    # from botorch.models.model_list_gp_regression import ModelListGP
    # from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    #
    # Y_train_standarized = standardize(y_train)
    # train_joint_YC = torch.cat([Y_train_standarized, c_train], dim=-1)
    #
    # models = []
    # for i in range(train_joint_YC.shape[-1]):
    #     models.append(
    #         SingleTaskGP(x_train, train_joint_YC[..., i: i + 1])
    #     )
    # model = ModelListGP(*models)
    # mll = SumMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_model(mll)
    #
    # posterior = model.posterior(plot_X)
    # mean = posterior.mean.detach().numpy()
    # is_feas = (mean[:,2] <= 0)
    # print("weights", weights)
    # mu_val_initial_conditions_raw = constrained_model.forward(plot_X.unsqueeze(dim=-2)).detach().numpy()
    #
    # import matplotlib.pyplot as plt
    # plt.scatter(mean[is_feas,0], mean[is_feas,1], c=mu_val_initial_conditions_raw.squeeze()[is_feas])
    #
    # Y_pareto_posterior = model.posterior(X_pareto_solutions)
    # Y_pareto_mean = Y_pareto_posterior.mean.detach().numpy()
    # print(Y_pareto_mean.shape)
    # plt.scatter(Y_pareto_mean[...,0], Y_pareto_mean[...,1], color="red")
    #
    # plt.show()
    # raise

    return X_pareto_solutions, X_pmean


def _compute_expected_utility(
        scalatization_fun: Callable,
        y_values: Tensor,
        c_values: Tensor,
        weights: Tensor,
) -> Tensor:
    utility = torch.zeros((weights.shape[0], y_values.shape[0]))

    for idx, w in enumerate(weights):
        scalarization = scalatization_fun(weights=w, Y=torch.Tensor([]).view((0, y_values.shape[1])))
        utility_values = scalarization(y_values).squeeze()
        utility[idx, :] = utility_values

    is_feas = (c_values <= 0).squeeze()
    if len(is_feas.shape) == 1:
        is_feas = is_feas.unsqueeze(dim=-2)

    aggregated_is_feas = torch.prod(is_feas, dim=1, dtype=bool)

    if aggregated_is_feas.sum() == 0:
        expected_utility = torch.Tensor([-100])
        return expected_utility
    else:
        utility_feas = utility[:, aggregated_is_feas]

        best_utility = torch.max(utility_feas, dim=1).values
        expected_utility = best_utility.mean()

        return expected_utility
