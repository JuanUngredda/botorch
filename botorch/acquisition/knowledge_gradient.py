#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch Knowledge Gradient (KG) via one-shot optimization as introduced in
[Balandat2020botorch]_. For broader discussion of KG see also [Frazier2008knowledge]_
and [Wu2016parallelkg]_.

.. [Balandat2020botorch]
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and
    E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
    Advances in Neural Information Processing Systems 33, 2020.

.. [Frazier2008knowledge]
    P. Frazier, W. Powell, and S. Dayanik. A Knowledge-Gradient policy for
    sequential information collection. SIAM Journal on Control and Optimization,
    2008.

.. [Wu2016parallelkg]
    J. Wu and P. Frazier. The parallel knowledge gradient method for batch
    bayesian optimization. NIPS 2016.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
from botorch import settings
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.analytic import DiscreteKnowledgeGradient
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.acquisition.objective import (
    AcquisitionObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor
from torch.distributions import Normal


class MCKnowledgeGradient(DiscreteKnowledgeGradient):
    r"""Knowledge Gradient using Monte-Carlo integration.

    This computes the Knowledge Gradient using randomly generated Zj ~ N(0,1) to
    find \max_{x'}{ \mu^n(x') + \hat{\sigma^n{x', x}}Z_{j} }. Then, the outer
    expectation is solved by taking the Monte Carlo average.
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: Optional[int] = 64,
        bounds: Tensor = None,
        inner_sampler: Optional[MCSampler] = None,
        objective: Optional[AcquisitionObjective] = None,
        seed: Optional[MCSampler] = 1,
        current_value: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            bounds: A `2 x d` tensor of lower and upper bounds for each column of
                the solutions to the inner problem.
            inner_sampler: The sampler used to sample fantasy observations.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
            kwargs: Additional keyword arguments. This includes the options for
                optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
                an `options` dictionary to be passed on to the optimization helpers, and
                a `scipy_options` dictionary to be passed to `scipy.minimize`.
        """

        if num_fantasies is None:
            raise ValueError("Must specify `num_fantasies`")

        super(MCKnowledgeGradient, self).__init__(
            model=model, bounds=bounds, num_discrete_points=num_fantasies
        )

        # This generates the fantasised samples according to a random seed.
        self.sampler = SobolQMCNormalSampler(
            num_samples=1, resample=True, collapse_batch_dims=False, seed=seed
        )

        self.bounds = bounds
        self.num_fantasies = num_fantasies
        self.current_value = current_value
        self.inner_sampler = inner_sampler
        self.objective = objective
        self.num_restarts = kwargs.get("num_restarts", 20)
        self.raw_samples = kwargs.get("raw_samples", 1024)
        self.kwargs = kwargs

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:

        kgvals = torch.zeros(X.shape[0], dtype=torch.double)
        for xnew_idx, xnew in enumerate(X):
            _, kgvals[xnew_idx] = self.compute_mc_kg(xnew=xnew, zvalues=None)

        return kgvals

    def compute_mc_kg(
        self, xnew: Tensor, zvalues: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            Zvals: 1 x num_fantasies Tensor with num_fantasies Normal quantiles.
        Returns:
            xstar_inner_optimisation: num_fantasies x d Tensor. Optimal X values for each z in Zvals.
            kg_estimated_value: 1 x 1 Tensor, Monte Carlo KG value of xnew
        """

        # There's a recurssion problem importing packages by one_shot kg. Hacky way of importing these packages.
        # TODO: find a better way to fix this
        if "gen_candidates_scipy" not in sys.modules:
            from botorch.generation.gen import gen_candidates_scipy
            from botorch.optim.initializers import gen_value_function_initial_conditions

        # Loop over xnew points
        fantasy_opt_val = torch.zeros((1, self.num_fantasies))  # 1 x num_fantasies
        xstar_inner_optimisation = torch.zeros((self.num_fantasies, xnew.shape[1]))

        # This setting makes sure that I can rewrite the base samples and use the quantiles.
        # Therefore, resample=False, collapse_batch_dims=True.
        sampler = SobolQMCNormalSampler(
            num_samples=1, resample=False, collapse_batch_dims=True
        )

        # loop over number of GP fantasised mean realisations
        for fantasy_idx in range(self.num_fantasies):

            # construct one realisation of the fantasy model by adding xnew. We rewrite the internal variable
            # base samples, such that the samples are taken from the quantile.
            if zvalues is None:
                fantasy_model = self.model.fantasize(
                    X=xnew, sampler=self.sampler, observation_noise=True
                )
            else:
                zval = zvalues[fantasy_idx].view(1, 1, 1)
                sampler.base_samples = zval

                fantasy_model = self.model.fantasize(
                    X=xnew, sampler=sampler, observation_noise=True
                )

            # get the value function and make sure gradients are enabled.
            with torch.enable_grad():
                value_function = _get_value_function(
                    model=fantasy_model,
                    objective=self.objective,
                    sampler=self.inner_sampler,
                    project=getattr(self, "project", None),
                )

                # optimize the inner problem
                initial_conditions = gen_value_function_initial_conditions(
                    acq_function=value_function,
                    bounds=self.bounds,
                    current_model=self.model,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    options = {
                              **self.kwargs.get("options", {}),
                              **self.kwargs.get("scipy_options", {}),
                          },
                )

                x_value, value = gen_candidates_scipy(
                    initial_conditions=initial_conditions,
                    acquisition_function=value_function,
                    lower_bounds=self.bounds[0],
                    upper_bounds=self.bounds[1],
                    options=self.kwargs.get("scipy_options"),
                )
                x_value = x_value # num initial conditions x 1 x d
                value = value.squeeze() # num_initial conditions

                # find top x in case there are several initial conditions
                x_top = x_value[torch.argmax(value)] # 1 x 1 x d

                # make sure to propagate kg gradients.
                with settings.propagate_grads(True):
                    x_top_val = value_function(X=x_top)

                fantasy_opt_val[:, fantasy_idx] = x_top_val
                xstar_inner_optimisation[fantasy_idx, :] = x_top.squeeze()
            # expectation computation

        kg_estimated_value = torch.mean(fantasy_opt_val, dim=-1)

        return xstar_inner_optimisation, kg_estimated_value


class HybridKnowledgeGradient(MCKnowledgeGradient):
    r"""Hybrid Knowledge Gradient using Monte-Carlo integration as described in
    Pearce M., Klaise J.,and Groves M. 2020. "Practical Bayesian Optimization
    of Objectives with Conditioning Variables". arXiv:2002.09996

    This acquisition function first computes high value design vectors using the
    predictive posterior GP mean for different Normal quantiles. Then discrete
    knowledge gradient is computed using the high value design vectors as a
    discretisation.
    """

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""

        Args:
            X: A `m x 1 x d` Tensor with `m` acquisition function evaluations of
            `d` dimensions. Currently DiscreteKnowledgeGradient does can't perform
            batched evaluations.

        Returns:
            kgvals: A 'm' Tensor with 'm' KG values.
        """

        """ compute hybrid KG """

        # generate equal quantile spaced z_vals
        zvalues = self.construct_z_vals(self.num_fantasies)

        kgvals = torch.zeros(X.shape[0], dtype=torch.double)
        for xnew_idx, xnew in enumerate(X):
            # Compute X discretisation using the different generated quantiles.
            x_star, _ = self.compute_mc_kg(xnew=xnew, zvalues=zvalues)

            # Compute value of discrete Knowledge Gradient using the generated discretisation
            kgvals[xnew_idx] = self.compute_discrete_kg(
                xnew=xnew, optimal_discretisation=x_star
            )

        return kgvals

    @staticmethod
    def construct_z_vals(nz: int, device: Optional[torch.device] = None) -> Tensor:
        """make nz equally quantile-spaced z values"""

        quantiles_z = (torch.arange(nz) + 0.5) * (1 / nz)
        normal = torch.distributions.Normal(0, 1)
        z_vals = normal.icdf(quantiles_z)
        return z_vals.to(device=device)


class qKnowledgeGradient(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""Batch Knowledge Gradient using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the inner
    expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models. For a fixed number
    of fantasies, all parts of `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: Optional[int] = 64,
        sampler: Optional[MCSampler] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None` or a ScalarizedObjective.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
        """
        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(
                num_samples=num_fantasies, resample=False, collapse_batch_dims=True
            )
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super(MCAcquisitionFunction, self).__init__(model=model)
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(
                num_samples=128, resample=False, collapse_batch_dims=True
            )
        if objective is None and model.num_outputs != 1:
            raise UnsupportedError(
                "Must specify an objective when using a multi-output model."
            )
        self.sampler = sampler
        self.objective = objective
        self.set_X_pending(X_pending)
        self.inner_sampler = inner_sampler
        self.num_fantasies = num_fantasies
        self.current_value = current_value

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """

        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=True
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model, objective=self.objective, sampler=self.inner_sampler
        )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        # return average over the fantasy samples
        return values.mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def evaluate(self, X: Tensor, bounds: Tensor, **kwargs: Any) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X_actual` by
        solving the inner optimization problem.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points
                each. Unlike `forward()`, this does not include solutions of the
                inner optimization problem.
            bounds: A `2 x d` tensor of lower and upper bounds for each column of
                the solutions to the inner problem.
            kwargs: Additional keyword arguments. This includes the options for
                optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
                an `options` dictionary to be passed on to the optimization helpers, and
                a `scipy_options` dictionary to be passed to `scipy.minimize`.

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X[b]` is averaged across the fantasy models.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X[b]`.
        """
        if hasattr(self, "expand"):
            X = self.expand(X)

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X, sampler=self.sampler, observation_noise=True
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model,
            objective=self.objective,
            sampler=self.inner_sampler,
            project=getattr(self, "project", None),
        )

        from botorch.generation.gen import gen_candidates_scipy

        # optimize the inner problem
        from botorch.optim.initializers import gen_value_function_initial_conditions

        initial_conditions = gen_value_function_initial_conditions(
            acq_function=value_function,
            bounds=bounds,
            num_restarts=kwargs.get("num_restarts", 20),
            raw_samples=kwargs.get("raw_samples", 1024),
            current_model=self.model,
            options={**kwargs.get("options", {}), **kwargs.get("scipy_options", {})},
        )

        _, values = gen_candidates_scipy(
            initial_conditions=initial_conditions,
            acquisition_function=value_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options=kwargs.get("scipy_options"),
        )
        # get the maximizer for each batch
        values, _ = torch.max(values, dim=0)
        if self.current_value is not None:
            values = values - self.current_value
        # NOTE: using getattr to cover both no-attribute with qKG and None with qMFKG
        if getattr(self, "cost_aware_utility", None) is not None:
            values = self.cost_aware_utility(
                X=X, deltas=values, sampler=self.cost_sampler
            )
        # return average over the fantasy samples
        return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_fantasies, :]


class qMultiFidelityKnowledgeGradient(qKnowledgeGradient):
    r"""Batch Knowledge Gradient for multi-fidelity optimization.

    A version of `qKnowledgeGradient` that supports multi-fidelity optimization
    via a `CostAwareUtility` and the `project` and `expand` operators. If none
    of these are set, this acquisition function reduces to `qKnowledgeGradient`.
    Through `valfunc_cls` and `valfunc_argfac`, this can be changed into a custom
    multifidelity acquisition function (it is only KG if the terminal value is
    computed using a posterior mean).
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: Optional[int] = 64,
        sampler: Optional[MCSampler] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None,
        cost_aware_utility: Optional[CostAwareUtility] = None,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        expand: Callable[[Tensor], Tensor] = lambda X: X,
        valfunc_cls: Optional[Type[AcquisitionFunction]] = None,
        valfunc_argfac: Optional[Callable[[Model, Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> None:
        r"""Multi-Fidelity q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None` or a ScalarizedObjective, then the analytic posterior mean
                is used, otherwise the objective is MC-evaluated (using
                inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None` or a ScalarizedObjective.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            project: A callable mapping a `batch_shape x q x d` tensor of design
                points to a tensor with shape `batch_shape x q_term x d` projected
                to the desired target set (e.g. the target fidelities in case of
                multi-fidelity optimization). For the basic case, `q_term = q`.
            expand: A callable mapping a `batch_shape x q x d` input tensor to
                a `batch_shape x (q + q_e)' x d`-dim output tensor, where the
                `q_e` additional points in each q-batch correspond to
                additional ("trace") observations.
            valfunc_cls: An acquisition function class to be used as the terminal
                value function.
            valfunc_argfac: An argument factory, i.e. callable that maps a `Model`
                to a dictionary of kwargs for the terminal value function (e.g.
                `best_f` for `ExpectedImprovement`).
        """
        if current_value is None and cost_aware_utility is not None:
            raise UnsupportedError(
                "Cost-aware KG requires current_value to be specified."
            )
        super().__init__(
            model=model,
            num_fantasies=num_fantasies,
            sampler=sampler,
            objective=objective,
            inner_sampler=inner_sampler,
            X_pending=X_pending,
            current_value=current_value,
        )
        self.cost_aware_utility = cost_aware_utility
        self.project = project
        self.expand = expand
        self._cost_sampler = None
        self.valfunc_cls = valfunc_cls
        self.valfunc_argfac = valfunc_argfac

    @property
    def cost_sampler(self):
        if self._cost_sampler is None:
            # Note: Using the deepcopy here is essential. Removing this poses a
            # problem if the base model and the cost model have a different number
            # of outputs or test points (this would be caused by expand), as this
            # would trigger re-sampling the base samples in the fantasy sampler.
            # By cloning the sampler here, the right thing will happen if the
            # the sizes are compatible, if they are not this will result in
            # samples being drawn using different base samples, but it will at
            # least avoid changing state of the fantasy sampler.
            self._cost_sampler = deepcopy(self.sampler)
        return self._cost_sampler

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiFidelityKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

                In addition, `X` may be augmented with fidelity parameteres as
                part of thee `d`-dimension. Projecting fidelities to the target
                fidelity is handled by `project`.

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_eval = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )
        else:
            X_eval = X_actual

        # construct the fantasy model of shape `num_fantasies x b`
        # expand X (to potentially add trace observations)
        fantasy_model = self.model.fantasize(
            X=self.expand(X_eval), sampler=self.sampler, observation_noise=True
        )
        # get the value function
        value_function = _get_value_function(
            model=fantasy_model,
            objective=self.objective,
            sampler=self.inner_sampler,
            project=self.project,
            valfunc_cls=self.valfunc_cls,
            valfunc_argfac=self.valfunc_argfac,
        )

        # make sure to propagate gradients to the fantasy model train inputs
        # project the fantasy points
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        if self.cost_aware_utility is not None:
            values = self.cost_aware_utility(
                X=X_actual, deltas=values, sampler=self.cost_sampler
            )

        # return average over the fantasy samples
        return values.mean(dim=0)


class ProjectedAcquisitionFunction(AcquisitionFunction):
    r"""
    Defines a wrapper around  an `AcquisitionFunction` that incorporates the project
    operator. Typically used to handle value functions in look-ahead methods.
    """

    def __init__(
        self,
        base_value_function: AcquisitionFunction,
        project: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(base_value_function.model)
        self.base_value_function = base_value_function
        self.project = project
        self.objective = base_value_function.objective
        self.sampler = getattr(base_value_function, "sampler", None)

    def forward(self, X: Tensor) -> Tensor:
        return self.base_value_function(self.project(X))


def _get_value_function(
    model: Model,
    objective: Optional[Union[MCAcquisitionObjective, ScalarizedObjective]] = None,
    sampler: Optional[MCSampler] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    valfunc_cls: Optional[Type[AcquisitionFunction]] = None,
    valfunc_argfac: Optional[Callable[[Model, Dict[str, Any]]]] = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    if valfunc_cls is not None:
        common_kwargs: Dict[str, Any] = {"model": model, "objective": objective}
        if issubclass(valfunc_cls, MCAcquisitionFunction):
            common_kwargs["sampler"] = sampler
        kwargs = valfunc_argfac(model=model) if valfunc_argfac is not None else {}
        base_value_function = valfunc_cls(**common_kwargs, **kwargs)
    else:
        if isinstance(objective, MCAcquisitionObjective):
            base_value_function = qSimpleRegret(
                model=model, sampler=sampler, objective=objective
            )

        else:
            base_value_function = PosteriorMean(model=model, objective=objective)

    if project is None:
        return base_value_function
    else:
        return ProjectedAcquisitionFunction(
            base_value_function=base_value_function,
            project=project,
        )


def _split_fantasy_points(X: Tensor, n_f: int) -> Tuple[Tensor, Tensor]:
    r"""Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x 1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy point
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x num_fantasies x d, needs to be num_fantasies x b x 1 x d
    # for batch mode evaluation with batch shape num_fantasies x b.
    # b x num_fantasies x d --> num_fantasies x b x d
    X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
    # num_fantasies x b x 1 x d
    X_fantasies = X_fantasies.unsqueeze(dim=-2)
    return X_actual, X_fantasies
