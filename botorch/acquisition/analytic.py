#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Normal

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Base class for analytic acquisition functions."""

    def __init__(
            self, model: Model, objective: Optional[ScalarizedObjective] = None
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            objective: A ScalarizedObjective (optional).
        """
        super().__init__(model=model)
        if objective is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify an objective when using a multi-output model."
                )
        elif not isinstance(objective, ScalarizedObjective):
            raise UnsupportedError(
                "Only objectives of type ScalarizedObjective are supported for "
                "analytic acquisition functions."
            )
        self.objective = objective

    def _get_posterior(self, X: Tensor) -> Posterior:
        r"""Compute the posterior at the input candidate set X.

        Applies the objective if provided.

        Args:
            X: The input candidate set.

        Returns:
            The posterior at X. If a ScalarizedObjective is defined, this
            posterior can be single-output even if the underlying model is a
            multi-output model.
        """
        posterior = self.model.posterior(X)
        if self.objective is not None:
            # Unlike MCAcquisitionObjective (which transform samples), this
            # transforms the posterior
            posterior = self.objective(posterior)
        return posterior

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )


class DiscreteKnowledgeGradient(AnalyticAcquisitionFunction):
    r"""Knowledge Gradient using a fixed discretisation in the Design Space "X"."""

    def __init__(
            self,
            model: Model,
            bounds: Optional[Tensor] = None,
            num_discrete_points: Optional[int] = None,
            X_discretisation: Optional[Tensor] = None,
            current_optimiser: Optional[Tensor] = None,
    ) -> None:
        r"""
        Discrete Knowledge Gradient
        Args:
            model: A fitted model.
            bounds: A `2 x d` tensor of lower and upper bounds for each column
            num_discrete_points: (int) The number of discrete points to use for input (X) space. More discrete
                points result in a better approximation, at the expense of
                memory and wall time.
            discretisation: A `k x d`-dim Tensor of `k` design points that will approximate the
                continuous space with a discretisation.
        """

        if X_discretisation is None:
            if num_discrete_points is None:
                raise ValueError(
                    "Must specify `num_discrete_points` for random discretisation if no `discretisation` is provided."
                )

            X_discretisation = draw_sobol_samples(
                bounds=bounds, n=num_discrete_points, q=1
            )

        super(AnalyticAcquisitionFunction, self).__init__(model=model)

        self.X_discretisation = X_discretisation
        self.current_optimiser = current_optimiser.squeeze()
    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        kgvals = torch.zeros(X.shape[0], dtype=torch.double)

        for xnew_idx, xnew in enumerate(X):
            xnew = xnew.unsqueeze(0)
            if self.current_optimiser is not None:
                self.current_optimiser = torch.atleast_2d(self.current_optimiser)
                self.X_discretisation = torch.cat([self.X_discretisation, self.current_optimiser.unsqueeze(dim=0)])

            kgvals[xnew_idx] = self.compute_discrete_kg(
                model=self.model, xnew=xnew, optimal_discretisation=self.X_discretisation
            )
        return kgvals

    @staticmethod
    def kgcb(a: Tensor, b: Tensor, plot=False) -> Tensor:
        r"""
        Calculates the linear epigraph, i.e. the boundary of the set of points
        in 2D lying above a collection of straight lines y=a+bx.
        Parameters
        ----------
        a
            Vector of intercepts describing a set of straight lines
        b
            Vector of slopes describing a set of straight lines
        Returns
        -------
        KGCB
            average height of the epigraph
        """

        a = a.squeeze()
        b = b.squeeze()
        a_old = a
        b_old = b
        assert len(a) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"

        maxa = torch.max(a)

        if torch.all(torch.abs(b) < 0.000000001):
            return torch.Tensor([0])  # , np.zeros(a.shape), np.zeros(b.shape)

        # Order by ascending b and descending a. There should be an easier way to do this
        # but it seems that pytorch sorts everything as a 1D Tensor

        ab_tensor = torch.vstack([-a, b]).T
        ab_tensor_sort_a = ab_tensor[ab_tensor[:, 0].sort()[1]]
        ab_tensor_sort_b = ab_tensor_sort_a[ab_tensor_sort_a[:, 1].sort()[1]]
        a = -ab_tensor_sort_b[:, 0]
        b = ab_tensor_sort_b[:, 1]

        # exclude duplicated b (or super duper similar b)
        threshold = (b[-1] - b[0]) * 0.00001
        diff_b = b[1:] - b[:-1]
        keep = diff_b > threshold
        keep = torch.cat([torch.Tensor([True]), keep])
        keep[torch.argmax(a)] = True
        keep = keep.bool()  # making sure 0 1's are transformed to booleans

        a = a[keep]
        b = b[keep]

        # initialize
        idz = [0]
        i_last = 0
        x = [-torch.inf]

        n_lines = len(a)
        # main loop TODO describe logic
        # TODO not pruning properly, e.g. a=[0,1,2], b=[-1,0,1]
        # returns x=[-inf, -1, -1, inf], shouldn't affect kgcb
        while i_last < n_lines - 1:
            i_mask = torch.arange(i_last + 1, n_lines)
            x_mask = -(a[i_last] - a[i_mask]) / (b[i_last] - b[i_mask])

            best_pos = torch.argmin(x_mask)
            idz.append(i_mask[best_pos])
            x.append(x_mask[best_pos])

            i_last = idz[-1]

        x.append(torch.inf)

        x = torch.Tensor(x)
        idz = torch.LongTensor(idz)
        # found the epigraph, now compute the expectation
        a = a[idz]
        b = b[idz]

        normal = Normal(torch.zeros_like(x), torch.ones_like(x))

        pdf = torch.exp(normal.log_prob(x))
        cdf = normal.cdf(x)

        kg = torch.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))
        kg -= maxa

        if plot:
            steps=20
            X = torch.linspace(start=-4.16 , end=4.16 , steps=steps, dtype=torch.double).unsqueeze(dim=-2)


            X_dims_adapted = torch.repeat_interleave(X, len(a_old), dim=0)
            b_old_plot = torch.repeat_interleave(b_old.unsqueeze(-1), steps, dim=1)
            a_old_plot = torch.repeat_interleave(a_old.unsqueeze(-1),steps, dim=1 )
            vals_old =  a_old_plot + b_old_plot * X_dims_adapted
            #
            import matplotlib.pyplot as plt
            plt.plot(X_dims_adapted.detach().numpy().T, vals_old.detach().numpy().T, color="grey", alpha=0.6)


            X_dims_adapted = torch.repeat_interleave(X, len(a), dim=0)
            b_plot = torch.repeat_interleave(b.unsqueeze(-1), steps, dim=1)
            a_plot = torch.repeat_interleave(a.unsqueeze(-1),steps, dim=1 )

            vals =  a_plot + b_plot * X_dims_adapted
            vals = torch.max(vals, dim=0).values


            if len(a_old)>60:
                plt.plot(X.detach().squeeze().numpy(), vals.detach().numpy(), label="X discrete")
            else:
                plt.plot(X.detach().squeeze().numpy(), vals.detach().numpy(), label="X optimised")

            plt.legend()
            # plt.show()


        return kg

    @staticmethod
    def compute_discrete_kg(
             model: Model, xnew: Tensor, optimal_discretisation: Tensor,
    plot=False, test=False) -> Tensor:
        """

        Args:
        xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            optimal_discretisation: num_fantasies x d Tensor. Optimal X values for each z in zvalues.

        """
        if test:

            DiscreteKnowledgeGradient.compute_discrete_kg(model=model, xnew=xnew,
                                                          optimal_discretisation=optimal_discretisation,
                                                          plot=True, test=False)

            dim = xnew.shape[-1]
            bounds_normalized = torch.vstack(
                [torch.zeros((1, dim)), torch.ones((1, dim))]
            )
            X_random_discretisation = draw_sobol_samples(
                bounds=bounds_normalized, n=100, q=1
            ).squeeze()
            xnew = torch.atleast_2d(xnew.squeeze())
            DiscreteKnowledgeGradient.compute_discrete_kg(model=model, xnew=xnew, optimal_discretisation=X_random_discretisation,
                                                          plot=True, test=False)

            import matplotlib.pyplot as plt
            plt.show()
        # Augment the discretisation with the designs.
        concatenated_xnew_discretisation = torch.cat(
            [xnew, optimal_discretisation], dim=0
        ).squeeze()  # (m + num_X_disc, d)

        # Compute posterior mean, variance, and covariance.
        full_posterior = model.posterior(
            concatenated_xnew_discretisation, observation_noise=False
        )
        noise_variance = torch.unique(model.likelihood.noise_covar.noise)
        full_posterior_mean = full_posterior.mean  # (1 + num_X_disc , 1)

        # Compute full Covariante Cov(Xnew, X_discretised), select [Xnew X_discretised] submatrix, and subvectors.
        full_posterior_covariance = (
            full_posterior.mvn.covariance_matrix
        )  # (1 + num_X_disc , 1 + num_X_disc )
        posterior_cov_xnew_opt_disc = full_posterior_covariance[
                                      : len(xnew), :
                                      ].squeeze()  # ( 1 + num_X_disc,)
        full_posterior_variance = (
            full_posterior.variance.squeeze()
        )  # (1 + num_X_disc, )

        full_predictive_covariance = (
                posterior_cov_xnew_opt_disc
                / (full_posterior_variance + noise_variance).sqrt()
        )
        # initialise empty kgvals torch.tensor

        kgval = DiscreteKnowledgeGradient.kgcb(a=full_posterior_mean, b=full_predictive_covariance, plot=plot)
        return kgval

class ExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei


class PosteriorMean(AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Mean.

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` property. The model must be single-outcome.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorMean(model)
        >>> pm = PM(test_X)
    """

    def __init__(
            self,
            model: Model,
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Posterior Mean.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem. Note
                that if `maximize=False`, the posterior mean is negated. As a
                consequence `optimize_acqf(PosteriorMean(gp, maximize=False))`
                does actually return -1 * minimum of the posterior mean.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given design
            points `X`.
        """
        posterior = self._get_posterior(X=X)
        mean = posterior.mean.view(X.shape[:-2])
        if self.maximize:
            return mean
        else:
            return -mean


class ProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Probability of Improvement.

    Probability of improvment over the current best observed value, computed
    using the analytic formula under a Normal posterior distribution. Only
    supports the case of q=1. Requires the posterior to be Gaussian. The model
    must be single-outcome.

    `PI(x) = P(y >= best_f), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PI = ProbabilityOfImprovement(model, best_f=0.2)
        >>> pi = PI(test_X)
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome analytic Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Probability of Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim tensor of Probability of Improvement values at the given
            design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        return normal.cdf(u)


class UpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return mean + delta
        else:
            return -mean + delta


class ConstrainedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Constrained Expected Improvement (feasibility-weighted).

    Computes the analytic expected improvement for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports the case `q=1`. The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.

    `Constrained_EI(x) = EI(x) * Product_i P(y_i \in [lower_i, upper_i])`,
    where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
    upper bounds for the i-th constraint, respectively.

    Example:
        >>> # example where 0th output has a non-negativity constraint and
        ... # 1st output is the objective
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> constraints = {0: (0.0, None)}
        >>> cEI = ConstrainedExpectedImprovement(model, 0.2, 1, constraints)
        >>> cei = cEI(test_X)
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            objective_index: int,
            constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
            maximize: bool = True,
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
        self.constraints = constraints
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self._preprocess_constraint_bounds(constraints=constraints)
        self.register_forward_pre_hook(convert_to_target_pre_hook)

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
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        means = posterior.mean.squeeze(dim=-2)  # (b) x m
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        # (b) x 1
        oi = self.objective_index
        mean_obj = means[..., oi: oi + 1]
        sigma_obj = sigmas[..., oi: oi + 1]
        u = (mean_obj - self.best_f.expand_as(mean_obj)) / sigma_obj
        if not self.maximize:
            u = -u
        normal = Normal(
            torch.zeros(1, device=u.device, dtype=u.dtype),
            torch.ones(1, device=u.device, dtype=u.dtype),
        )
        ei_pdf = torch.exp(normal.log_prob(u))  # (b) x 1
        ei_cdf = normal.cdf(u)
        ei = sigma_obj * (ei_pdf + u * ei_cdf)
        prob_feas = self._compute_prob_feas(X=X, means=means, sigmas=sigmas)
        ei = ei.mul(prob_feas)
        return ei.squeeze(dim=-1)

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


class NoisyExpectedImprovement(ExpectedImprovement):
    r"""Single-outcome Noisy Expected Improvement (via fantasies).

    This computes Noisy Expected Improvement by averaging over the Expected
    Improvemnt values of a number of fantasy models. Only supports the case
    `q=1`. Assumes that the posterior distribution of the model is Gaussian.
    The model must be single-outcome.

    `NEI(x) = E(max(y - max Y_baseline), 0)), (y, Y_baseline) ~ f((x, X_baseline))`,
    where `X_baseline` are previously observed points.

    Note: This acquisition function currently relies on using a FixedNoiseGP (required
    for noiseless fantasies).

    Example:
        >>> model = FixedNoiseGP(train_X, train_Y, train_Yvar=train_Yvar)
        >>> NEI = NoisyExpectedImprovement(model, train_X)
        >>> nei = NEI(test_X)
    """

    def __init__(
            self,
            model: GPyTorchModel,
            X_observed: Tensor,
            num_fantasies: int = 20,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Noisy Expected Improvement (via fantasies).

        Args:
            model: A fitted single-outcome model.
            X_observed: A `n x d` Tensor of observed points that are likely to
                be the best observed points so far.
            num_fantasies: The number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance).
            maximize: If True, consider the problem a maximization problem.
        """
        if not isinstance(model, FixedNoiseGP):
            raise UnsupportedError(
                "Only FixedNoiseGPs are currently supported for fantasy NEI"
            )
        # sample fantasies
        with torch.no_grad():
            posterior = model.posterior(X=X_observed)
            sampler = SobolQMCNormalSampler(num_fantasies)
            Y_fantasized = sampler(posterior).squeeze(-1)
        batch_X_observed = X_observed.expand(num_fantasies, *X_observed.shape)
        # The fantasy model will operate in batch mode
        fantasy_model = _get_noiseless_fantasy_model(
            model=model, batch_X_observed=batch_X_observed, Y_fantasized=Y_fantasized
        )

        if maximize:
            best_f = Y_fantasized.max(dim=-1)[0]
        else:
            best_f = Y_fantasized.min(dim=-1)[0]

        super().__init__(model=fantasy_model, best_f=best_f, maximize=maximize)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `b1 x ... bk`-dim tensor of Noisy Expected Improvement values at
            the given design points `X`.
        """
        # add batch dimension for broadcasting to fantasy models
        return super().forward(X.unsqueeze(-3)).mean(dim=-1)


def _construct_dist(means: Tensor, sigmas: Tensor, inds: Tensor) -> Normal:
    mean = means.index_select(dim=-1, index=inds)
    sigma = sigmas.index_select(dim=-1, index=inds)
    return Normal(loc=mean, scale=sigma)


def _get_noiseless_fantasy_model(
        model: FixedNoiseGP, batch_X_observed: Tensor, Y_fantasized: Tensor
) -> FixedNoiseGP:
    r"""Construct a fantasy model from a fitted model and provided fantasies.

    The fantasy model uses the hyperparameters from the original fitted model and
    assumes the fantasies are noiseless.

    Args:
        model: a fitted FixedNoiseGP
        batch_X_observed: A `b x n x d` tensor of inputs where `b` is the number of
            fantasies.
        Y_fantasized: A `b x n` tensor of fantasized targets where `b` is the number of
            fantasies.

    Returns:
        The fantasy model.
    """
    # initialize a copy of FixedNoiseGP on the original training inputs
    # this makes FixedNoiseGP a non-batch GP, so that the same hyperparameters
    # are used across all batches (by default, a GP with batched training data
    # uses independent hyperparameters for each batch).
    fantasy_model = FixedNoiseGP(
        train_X=model.train_inputs[0],
        train_Y=model.train_targets.unsqueeze(-1),
        train_Yvar=model.likelihood.noise_covar.noise.unsqueeze(-1),
    )
    # update training inputs/targets to be batch mode fantasies
    fantasy_model.set_train_data(
        inputs=batch_X_observed, targets=Y_fantasized, strict=False
    )
    # use noiseless fantasies
    fantasy_model.likelihood.noise_covar.noise = torch.full_like(Y_fantasized, 1e-7)
    # load hyperparameters from original model
    state_dict = deepcopy(model.state_dict())
    fantasy_model.load_state_dict(state_dict)
    return fantasy_model


class ScalarizedPosteriorMean(AnalyticAcquisitionFunction):
    r"""Scalarized Posterior Mean.

    This acquisition function returns a scalarized (across the q-batch)
    posterior mean given a vector of weights.
    """

    def __init__(
            self,
            model: Model,
            weights: Tensor,
            objective: Optional[ScalarizedObjective] = None,
    ) -> None:
        r"""Scalarized Posterior Mean.

        Args:
            model: A fitted single-outcome model.
            weights: A tensor of shape `q` for scalarization.
            objective: A ScalarizedObjective. Required for multi-output models.
        """
        super().__init__(model=model, objective=objective)
        self.register_buffer("weights", weights.unsqueeze(dim=0))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the scalarized posterior mean on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given design
            points `X`.
        """
        posterior = self._get_posterior(X=X)
        weighted_means = posterior.mean.squeeze(dim=-1) * self.weights
        return weighted_means.sum(dim=-1)
