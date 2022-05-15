#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from __future__ import annotations

from typing import Callable, Optional
from typing import Dict, Tuple, Any

import torch
from torch import Tensor

from botorch import settings
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _construct_dist
from botorch.acquisition.multi_objective.monte_carlo import (
    MultiObjectiveMCAcquisitionFunction,
)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
)
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.converter import (
    model_list_to_batched,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import (
    t_batch_mode_transform,
)



class MultiAttributeConstrainedKG(MultiObjectiveMCAcquisitionFunction):
    r"""Abstract base class for MC multi-output objectives."""

    def __init__(self,
                 model: Model,
                 bounds: Tensor,
                 utility_model: Callable,
                 num_objectives: int,
                 num_fantasies: int,
                 num_scalarisations: int,
                 sampler: Optional[MCSampler] = None,
                 objective: Optional[MCMultiOutputObjective] = None,
                 X_pending: Optional[Tensor] = None,
                 **kwargs: Any) -> None:

        super().__init__(model=model,
                         sampler=sampler,
                         objective=objective,
                         X_pending=X_pending)

        self.utility_model = utility_model
        self.num_fantasies = num_fantasies
        self.input_dim = model.train_inputs[0][0].shape[1]
        self.num_outputs = model.num_outputs
        self.num_objectives = num_objectives
        self.num_scalarisations = num_scalarisations
        self.num_X_observations = None
        self.bounds = bounds

        self.num_restarts = kwargs.get("num_restarts", 1)
        self.raw_samples = kwargs.get("raw_samples", 100)
        self.optional = {"num_restarts": self.num_restarts, "raw_samples": self.raw_samples}
        # convert to batched MO model
        batched_mo_model = (
            model_list_to_batched(model) if isinstance(model, ModelListGP) else model
        )
        self._init_model = batched_mo_model
        self.mo_model = batched_mo_model

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        r"""Evaluate the multi-output objective on the samples.

        Args:
            X: A `batch_shape x q x d`-dim Tensors of inputs.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim Tensor of objective values with
            `m'` the output dimension. This assumes maximization in each output
            dimension).

        This method is usually not called directly, but via the objectives
        """

        X_argmax_pmean, weights, zvalues = self._initialize_maKG_parameters(model=self.model)
        kgvals = torch.zeros(X.shape[0], dtype=torch.double)
        for xnew_idx, xnew in enumerate(X):
            kgvals[xnew_idx] = self._compute_mackg(model=self.model,
                                                   xnew=xnew,
                                                   weights=weights,
                                                   zvalues=zvalues,
                                                   X_current_argmax_pmean=X_argmax_pmean)

        return kgvals

    def _compute_mackg(self,
                       model: Model,
                       xnew: Tensor,
                       weights: Tensor,
                       zvalues: Tensor,
                       X_current_argmax_pmean: Tensor) -> tuple[Tensor, Tensor]:

        # Loop over xnew points
        fantasy_opt_val = torch.zeros((self.num_scalarisations, self.num_fantasies))  # 1 x num_fantasies

        posterior = model.posterior(self.dummy_X)
        dummy_mean = posterior.mean[..., :self.num_objectives]

        for w_idx, w_i in enumerate(weights):
            scalarization = self.utility_model(weights=w_i, Y=dummy_mean)

            sampler = SobolQMCNormalSampler(
                num_samples=1, resample=False, collapse_batch_dims=True
            )

            # xstar_inner_optimisation = torch.zeros((self.num_fantasies, xnew.shape[1]))

            # loop over number of GP fantasised mean realisations
            for fantasy_idx in range(self.num_fantasies):
                # construct one realisation of the fantasy model by adding xnew. We rewrite the internal variable
                # base samples, such that the samples are taken from the quantile.
                zval = zvalues[fantasy_idx].view(1, 1, self.num_outputs)
                sampler.base_samples = zval

                # fantasize the model
                fantasy_model = self.mo_model.fantasize(
                    X=xnew, sampler=sampler, observation_noise=True
                )

                constrained_model = ConstrainedPosteriorMean(
                    model=fantasy_model,
                    objective_index=self.num_objectives,
                    scalarization=scalarization,
                )
                x_current_max = X_current_argmax_pmean[w_idx].unsqueeze(dim=-2)
                x_top, x_top_val = self.argmax_constrained_model(value_fun=constrained_model,
                                                                 batch_initial_conditions=X_current_argmax_pmean.unsqueeze(
                                                                     dim=-2))

                current_max_val = constrained_model.forward(x_current_max.unsqueeze(dim=-2))

                # if len(self.hashmap)==0:
                #
                #     x_current_max = X_current_argmax_pmean[w_idx].unsqueeze(dim=-2)
                #     x_top, x_top_val = self.argmax_constrained_model(value_fun=constrained_model,
                #                                                      batch_initial_conditions=X_current_argmax_pmean.unsqueeze(dim=-2))
                #
                #     current_max_val = constrained_model.forward(x_current_max.unsqueeze(dim=-2))
                #
                #     model_values_test = constrained_model.forward(self.X_discretisation_test).detach().squeeze()
                #     self.model_values_test_dict[0] = model_values_test
                #     self.hashmap[0] = [current_max_val, x_top_val]
                # else:
                #     model_values_test = constrained_model.forward(self.X_discretisation_test).detach().squeeze()
                #     # print("model_vals", model_values_test)
                #     found_solutions = False
                #     for k in self.model_values_test_dict.keys():
                #         # print("self.model_values_test_dict[k]",self.model_values_test_dict[k])
                #         # print("overall metric", torch.sum(torch.abs(self.model_values_test_dict[k] - model_values_test)))
                #         if torch.sum(torch.abs(self.model_values_test_dict[k] - model_values_test)) < 1e-8:
                #
                #             current_max_val = self.hashmap[k][0]
                #             x_top_val = self.hashmap[k][1]
                #             found_solutions = True
                #             print("1")
                #     if found_solutions is False:
                #         x_current_max = X_current_argmax_pmean[w_idx].unsqueeze(dim=-2)
                #         x_top, x_top_val = self.argmax_constrained_model(value_fun=constrained_model,
                #                                                          batch_initial_conditions=X_current_argmax_pmean.unsqueeze(
                #                                                              dim=-2))
                #
                #         current_max_val = constrained_model.forward(x_current_max.unsqueeze(dim=-2))
                #
                #         Kth_idx = list(self.model_values_test_dict.keys())[-1]
                #         self.model_values_test_dict[Kth_idx + 1] = model_values_test
                #         self.hashmap[Kth_idx + 1] = [current_max_val, x_top_val]



                # if  True:#x_top_val - current_max_val < 0:
                #     print("weight", w_i)
                #     X_plot = torch.rand((1000, 1, 1, self.input_dim))
                #     constrained_posterior_mean = xnew_constrained_model.forward(
                #         X=X_plot)  # number_X_inner x number_realisation_xnew
                #     constrained_posterior_mean = constrained_posterior_mean.detach().numpy()
                #
                #     posterior = fantasy_model.posterior(X=X_plot)
                #     means = posterior.mean.squeeze(dim=-2)
                #     oi = self.num_objectives
                #     mean_obj = means[..., :oi].detach().numpy()
                #
                #     print(mean_obj.shape)
                #     print(constrained_posterior_mean.shape, constrained_posterior_mean.max(),
                #           constrained_posterior_mean.min())
                #
                #     import matplotlib.pyplot as plt
                #     posterior = fantasy_model.posterior(X=x_top)
                #     means = posterior.mean.squeeze(dim=-2)
                #     oi = self.num_objectives
                #     top_mean_obj = means[..., :oi].squeeze().detach().numpy()
                #
                #     posterior = fantasy_model.posterior(X=x_current_max.unsqueeze(dim=-2))
                #     means = posterior.mean.squeeze(dim=-2)
                #     oi = self.num_objectives
                #     current_mean_obj = means[..., :oi].squeeze().detach().numpy()
                #     for i in range(mean_obj.shape[1]):
                #         plt.scatter(mean_obj[:, i, 0], mean_obj[:, i, 1], c=constrained_posterior_mean[:, i].reshape(-1))
                #         plt.scatter(top_mean_obj[0], top_mean_obj[1], color="red")
                #         plt.scatter(current_mean_obj[0], current_mean_obj[1], color="red", marker="x")
                #         plt.show()
                #     raise
                fantasy_opt_val[w_idx, fantasy_idx] = x_top_val - current_max_val
                # xstar_inner_optimisation[fantasy_idx, :] = x_top.squeeze()

            fantasy_opt_val = fantasy_opt_val.clamp_min(1e-9)

            return fantasy_opt_val.mean()


    def argmax_constrained_model(self, value_fun: AnalyticAcquisitionFunction, batch_initial_conditions: Tensor):
        # get the value function and make sure gradients are enabled.

        # optimize the inner problem

        with torch.enable_grad():
            domain_offset = self.bounds[0]
            domain_range = self.bounds[1] - self.bounds[0]
            X_unit_cube_samples = torch.rand((self.raw_samples, 1, 1, self.input_dim))
            X_initial_conditions_raw = X_unit_cube_samples * domain_range + domain_offset
            X_initial_conditions_raw = torch.cat([X_initial_conditions_raw, batch_initial_conditions])

            mu_val_initial_conditions_raw = value_fun.forward(X_initial_conditions_raw)
            best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                             :self.num_restarts].squeeze()

            X_initial_conditions = X_initial_conditions_raw[best_k_indeces, ...]


            x_value, value = gen_candidates_scipy(
                initial_conditions=X_initial_conditions.unsqueeze(dim=-2),
                acquisition_function=value_fun,
                lower_bounds=self.bounds[0],
                upper_bounds=self.bounds[1])

        x_value = x_value  # num_initial conditions x 1 x d
        value = value.squeeze()  # num_initial conditions
        # print("value", value)

        # find top x in case there are several initial conditions
        x_top = x_value[torch.argmax(value)]  # 1 x 1 x d

        # make sure to propagate kg gradients.
        with settings.propagate_grads(True):
            x_top_val = value_fun(X=x_top.unsqueeze(dim=-2))

        return x_top, x_top_val

    def _initialize_maKG_parameters(self, model: Model):

        current_number_of_observations = self.model.train_inputs[0][0].shape[0]

        # Z values are only updated if new data is included in the model.
        # This ensures that we can use a deterministic optimizer.
        bounds_normalized = torch.vstack([torch.zeros(self.input_dim), torch.ones(self.input_dim)])

        if current_number_of_observations != self.num_X_observations:
            # sample random weights
            self.dummy_X = torch.rand((1000, self.input_dim))
            weights = sample_simplex(
                n=self.num_scalarisations, d=self.num_objectives
            ).squeeze()

            X_pareto_solutions, _ = ParetoFrontApproximation(
                model=model,
                scalatization_fun=self.utility_model,
                input_dim=self.input_dim,
                bounds=bounds_normalized,
                dummy_X=self.dummy_X,
                num_objectives=self.num_objectives,
                weights=weights,
                optional=self.optional,
            )

            z_vals = draw_sobol_normal_samples(
                d=self.num_outputs,
                n=self.num_fantasies
            ).squeeze()  # 1 x num_fantasies
            self.z_vals = z_vals
            self.weights = weights
            self.X_pareto_solutions = X_pareto_solutions
            self.num_X_observations = current_number_of_observations
            # self.hashmap = {}
            # self.model_values_test_dict = {}
            # self.X_discretisation_test =  torch.rand((1000, 1, 1, self.input_dim))

        else:
            weights = self.weights
            z_vals = self.z_vals
            X_pareto_solutions = self.X_pareto_solutions

        return X_pareto_solutions, weights, z_vals




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
        assert len(X.shape) == 4, "dim issue {}".format(X.shape)


        posterior = self._get_posterior(X=X)
        means = posterior.mean.squeeze(dim=-2)  # (b) x m
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        # (b) x 1
        oi = self.objective_index
        mean_obj = means[..., :oi]

        scalarized_objective = self.scalarization(mean_obj)
        mean_constraints = means[..., oi:].squeeze(dim=-2)
        sigma_constraints = sigmas[..., oi:].squeeze(dim=-2)

        # print("X.squeeze(dim=-2)", X.squeeze(dim=-2).shape)
        # print("mean_constraints", mean_constraints.shape)
        prob_feas = self._compute_prob_feas(
            X=X.squeeze(dim=-2),
            means=mean_constraints,
            sigmas=sigma_constraints,
        )
        constrained_posterior_mean = scalarized_objective.mul(prob_feas)
        return constrained_posterior_mean

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

def construct_z_vals( nz: int, device: Optional[torch.device] = None) -> Tensor:
    """make nz equally quantile-spaced z values"""

    quantiles_z = (torch.arange(nz) + 0.5) * (1 / nz)
    normal = torch.distributions.Normal(0, 1)
    z_vals = normal.icdf(quantiles_z)
    return z_vals.to(device=device)

def ParetoFrontApproximation(
        model: Model,
        input_dim: int,
        scalatization_fun: Callable,
        bounds: Tensor,
        num_objectives: int,
        weights: Tensor,
        dummy_X: Optional[Tensor] = None,
        optional: Optional[dict[str, int]] = None,
) -> tuple[Tensor, Tensor]:
    X_pareto_solutions = []
    X_pmean = []

    if dummy_X is None:
        dummy_X = torch.rand((500, input_dim))

    posterior = model.posterior(dummy_X)
    dummy_mean = posterior.mean[..., :num_objectives]

    for w in weights:
        # normalizes scalarization between [0,1] given the training data.
        scalarization = scalatization_fun(weights=w, Y=dummy_mean)

        constrained_model = ConstrainedPosteriorMean(
            model=model,
            objective_index=num_objectives,
            scalarization=scalarization,
        )

        X_initial_conditions_raw = torch.rand((optional["raw_samples"], 1, 1, input_dim))

        mu_val_initial_conditions_raw = constrained_model.forward(
            X_initial_conditions_raw
        )

        best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                         : optional["num_restarts"]
                         ].squeeze()

        with torch.enable_grad():

            X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

            top_x_initial_means, value_initial_means = gen_candidates_scipy(
                initial_conditions=X_initial_conditions.unsqueeze(dim=-2),
                acquisition_function=constrained_model,
                lower_bounds=torch.zeros(input_dim),
                upper_bounds=torch.ones(input_dim),
            )

        # subopt_x = X_initial_conditions[torch.argmax(value_initial_means), ...]
        top_x = top_x_initial_means[torch.argmax(value_initial_means), ...]
        X_pareto_solutions.append(top_x)
        X_pmean.append(torch.max(value_initial_means))

        # X_random_start.append(subopt_x)
        # plot_X = torch.rand((1000, 4))
        # posterior = model.posterior(plot_X)
        # mean = posterior.mean.detach().numpy()
        # is_feas = (mean[:, 2] <= 0)
        # print("weight", weights[0])
        # import matplotlib.pyplot as plt
        # plt.scatter(mean[is_feas, 0], mean[is_feas, 1], c=mean[is_feas, 2])
        #
        # Y_pareto_posterior = model.posterior(top_x)
        # Y_pareto_mean = Y_pareto_posterior.mean.detach().numpy()
        # print(Y_pareto_mean.shape)
        # plt.scatter(Y_pareto_mean[..., 0], Y_pareto_mean[..., 1], color="red")
        # plt.show()
        # raise
    X_pareto_solutions = torch.vstack(X_pareto_solutions)
    X_pmean = torch.vstack(X_pmean)

    # X_random_start = torch.vstack(X_random_start)

    # plot_X = torch.rand((1000,3))
    # posterior = model.posterior(plot_X)
    # mean = posterior.mean.detach().numpy()
    # is_feas = (mean[:,2] <= 0)
    # print("mean", mean.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(mean[is_feas,0], mean[is_feas,1], c=mean[is_feas,2])
    #
    # Y_pareto_posterior = model.posterior(X_pareto_solutions)
    # Y_pareto_mean = Y_pareto_posterior.mean.detach().numpy()
    # print(Y_pareto_mean.shape)
    # plt.scatter(Y_pareto_mean[...,0], Y_pareto_mean[...,1], color="red")
    #
    # plt.show()
    # raise
    # raise
    return X_pareto_solutions, X_pmean
