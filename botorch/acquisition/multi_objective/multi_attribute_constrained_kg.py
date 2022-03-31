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
from torch.distributions import Normal

from botorch.acquisition.analytic import _construct_dist
from botorch.acquisition.multi_objective.monte_carlo import (
    MultiObjectiveMCAcquisitionFunction,
)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
)
from botorch.models.converter import (
    model_list_to_batched,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import t_batch_mode_transform


class MultiAttributeConstrainedKG(MultiObjectiveMCAcquisitionFunction):
    r"""Abstract base class for MC multi-output objectives."""

    def __init__(self,
                 model: Model,
                 bounds: Tensor,
                 utility_model: Callable,
                 num_objectives: int,
                 num_fantasies_constraints: int,
                 X_discretisation_size: int,
                 num_scalarisations: int,
                 current_global_optimiser = Tensor,
                 sampler: Optional[MCSampler] = None,
                 fixed_scalarizations: Optional[Tensor] = None,
                 objective: Optional[MCMultiOutputObjective] = None,
                 X_pending: Optional[Tensor] = None,
                 **kwargs: Any) -> None:

        super().__init__(model=model,
                         sampler=sampler,
                         objective=objective,
                         X_pending=X_pending)

        self.utility_model = utility_model
        self.num_fantasies_constraints = num_fantasies_constraints
        self.input_dim = model.train_inputs[0][0].shape[1]
        self.num_outputs = model.num_outputs
        self.num_objectives = fixed_scalarizations.shape[0]
        self.num_constraints = model.num_outputs - self.num_objectives
        self.X_discretisation_size = X_discretisation_size
        self.num_scalarisations = num_scalarisations
        self.num_X_observations = None
        self.bounds = bounds
        self.fixed_scalarizations = fixed_scalarizations
        self.name = "MultiAttributeConstrainedKG"
        self.num_restarts = kwargs.get("num_restarts", 1)
        self.raw_samples = kwargs.get("raw_samples", 100)
        self.optional = {"num_restarts": self.num_restarts, "raw_samples": self.raw_samples}
        self.X_discretisation = draw_sobol_samples(bounds=bounds, n=100, q=1)
        self.current_global_optimiser = current_global_optimiser.squeeze(dim=-2)
        if self.X_pending is not None:
            self.X_discretisation = torch.concat([self.X_discretisation, self.X_pending]).squeeze()
        # convert to batched MO model
        self.model_obj = model.subset_output(idcs=range(self.num_objectives))
        self.model_cs = model.subset_output(idcs=range(self.num_objectives, model.num_outputs))

        batched_cs_model = (
            model_list_to_batched(self.model_cs) if isinstance(self.model_cs, ModelListGP) else self.model_cs
        )

        self.mo_cs_model = batched_cs_model

        default_value = (None, 0)
        constraints = dict.fromkeys(
            range(model.num_outputs - self.num_objectives), default_value
        )
        self._preprocess_constraint_bounds(constraints=constraints)

    @t_batch_mode_transform()
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
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.X_discretisation_size)

        zvalues = self._initialize_maKG_parameters(model=self.model)
        kgvals = torch.zeros(X.shape[0], dtype=torch.double)

        for x_i, xnew in enumerate(X_actual):
            X_discretisation = X_fantasies[:, x_i, ...].squeeze()
            X_discretisation = torch.cat([X_discretisation, self.current_global_optimiser])
            kgvals[x_i] = self._compute_mackg(
                xnew=xnew,
                weights=self.fixed_scalarizations,
                zvalues=zvalues,
                optimal_discretisation=X_discretisation)

        return kgvals

    def _compute_mackg(self,
                       xnew: Tensor,
                       weights: Tensor,
                       zvalues: Tensor,
                       optimal_discretisation: Tensor) -> tuple[Tensor, Tensor]:

        # Loop over xnew points

        fantasy_opt_val = torch.zeros((self.num_scalarisations, self.num_fantasies_constraints))  # 1 x num_fantasies

        for w_idx, _ in enumerate(weights):

            sampler = SobolQMCNormalSampler(
                num_samples=1, resample=False, collapse_batch_dims=True
            )

            # loop over number of GP fantasised mean realisations
            for fantasy_idx in range(zvalues.shape[0]):

                # construct one realisation of the fantasy model by adding xnew. We rewrite the internal variable
                # base samples, such that the samples are taken from the quantile.
                zval = zvalues[fantasy_idx, :].view(1, 1, self.num_constraints)
                sampler.base_samples = zval

                # fantasize the model
                fantasy_model_cs = self.mo_cs_model.fantasize(
                    X=xnew, sampler=sampler, observation_noise=True
                )

                model_obj = self.model_obj.subset_output(idcs=[w_idx])

                discKG = self.compute_discrete_kg(xnew=xnew,
                                                  model_obj=model_obj,
                                                  model_cs=fantasy_model_cs,
                                                  optimal_discretisation=optimal_discretisation)

                fantasy_opt_val[w_idx, fantasy_idx] = discKG

        return fantasy_opt_val.mean()

    def _initialize_maKG_parameters(self, model: Model):

        current_number_of_observations = self.model.train_inputs[0][0].shape[0]

        # Z values are only updated if new data is included in the model.
        # This ensures that we can use a deterministic optimizer.
        bounds_normalized = torch.vstack([torch.zeros(self.input_dim), torch.ones(self.input_dim)])

        if current_number_of_observations != self.num_X_observations:

            num_constraints = self.num_outputs - self.num_objectives
            constraint_base_zvals = construct_z_vals(nz=self.num_fantasies_constraints)

            z_vals = torch.atleast_2d(constraint_base_zvals).T

            self.z_vals = z_vals
            self.num_X_observations = current_number_of_observations

            assert z_vals.shape[0] == self.num_fantasies_constraints, "num of z vals is not the same as num of fantasies"
            assert z_vals.shape[1] == self.num_constraints, "there should be a zval per dimension"
        else:

            z_vals = self.z_vals

        return z_vals

    def compute_discrete_kg(
            self, xnew: Tensor, model_obj: Model, model_cs: Model, optimal_discretisation: Tensor
    ) -> Tensor:
        """

        Args:
        xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            optimal_discretisation: num_fantasies x d Tensor. Optimal X values for each z in zvalues.

        """

        # Augment the discretisation with the designs.
        concatenated_xnew_discretisation = torch.cat(
            [xnew, optimal_discretisation], dim=0
        ).squeeze()  # (m + num_X_disc, d)

        # Compute posterior mean, variance, and covariance.
        full_posterior = model_obj.posterior(
            concatenated_xnew_discretisation, observation_noise=False
        )
        noise_variance = torch.unique(model_obj.likelihood.likelihoods[0].noise_covar.noise)
        full_posterior_mean = full_posterior.mean.squeeze()  # (1 + num_X_disc , 1)

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
        ).squeeze()

        posterior_cs = model_cs.posterior(X=concatenated_xnew_discretisation)
        mean_constraints = posterior_cs.mean.squeeze(dim=-2)  # (b) x m
        sigma_constraints = posterior_cs.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        prob_feas = self._compute_prob_feas(
            X=concatenated_xnew_discretisation.squeeze(dim=-2),
            means=mean_constraints.squeeze(dim=-2),
            sigmas=sigma_constraints.squeeze(dim=-2),
        ).double().squeeze()

        # initialise empty kgvals torch.tensor
        kgval = self.kgcb(a=full_posterior_mean * prob_feas, b=full_predictive_covariance * prob_feas)

        return kgval

    def _plot(self, X,
              X_train,
              Y_train,
              C_train,
              true_fun):

        x_best, optimal_discretisation = _split_fantasy_points(X=X, n_f=self.X_discretisation_size)

        plot_X = torch.rand((1000, 1, 3))

        from botorch.fit import fit_gpytorch_model
        from botorch.models import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
        from botorch.utils import standardize
        import matplotlib.pyplot as plt
        from botorch.utils.transforms import unnormalize

        Y_train_standarized = Y_train#standardize(Y_train)
        train_joint_YC = torch.cat([Y_train_standarized, C_train], dim=-1)

        models = []
        for i in range(train_joint_YC.shape[-1]):
            models.append(
                SingleTaskGP(X_train, train_joint_YC[..., i: i + 1])
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        with torch.no_grad():

            x = unnormalize(X=plot_X, bounds=self.bounds)
            objective = torch.vstack([true_fun(x_i) for x_i in x]).to(dtype =torch.double)
            constraints = -torch.vstack([true_fun.evaluate_slack(x_i) for x_i in x]).to(dtype =torch.double)
            is_feas = (constraints.squeeze() <= 0)

            posterior = model.posterior(plot_X)
            mean = posterior.mean.squeeze().detach().numpy()
            # is_feas = (mean[..., -1] <= 0)

            posterior_best = model.posterior(x_best)
            mean_best = posterior_best.mean.squeeze().detach().numpy()

            # plt.scatter(mean[:, 0], mean[:, 1], color="magenta", alpha=0.3, s=20)
            # plt.scatter(mean[is_feas, 0], mean[is_feas, 1], color="blue", alpha=0.3, s=20)  # , c=mean[is_feas, 2])
            # plt.scatter(mean_best[0], mean_best[1], color="red")
            # plt.scatter(Y_train_standarized.squeeze()[:,0],Y_train_standarized.squeeze()[:,1], color="orange")
            # plt.show()
            # plt.cla()

            zvalues = self._initialize_maKG_parameters(model=self.model)
            kgvals = torch.zeros(plot_X.shape[0], dtype=torch.double)

            # for zi in zvalues:
            #
            #     for x_i, xnew in enumerate(plot_X):
            #         kgvals[x_i] = self._compute_mackg(
            #             xnew=xnew,
            #             weights=self.fixed_scalarizations,
            #             zvalues=torch.atleast_2d(zi),
            #             optimal_discretisation=optimal_discretisation.squeeze())
            #
            #     print("mean",torch.mean(kgvals),"max", torch.max(kgvals),"min", torch.min(kgvals))
            #     plt.title(str(zi))
            #     plt.scatter(mean[:, 0], mean[:, 1], c=kgvals)
            #     plt.scatter(mean_best[0], mean_best[1], color="red")
            #     plt.show()

            for x_i, xnew in enumerate(plot_X):
                kgvals[x_i] = self._compute_mackg(
                    xnew=xnew,
                    weights=self.fixed_scalarizations,
                    zvalues=zvalues,
                    optimal_discretisation=optimal_discretisation.squeeze())

            print("mean", torch.mean(kgvals), "max", torch.max(kgvals), "min", torch.min(kgvals))
            plt.scatter(objective[is_feas,0], objective[is_feas,1], color="green")
            plt.scatter(mean[:, 0], mean[:, 1], c=kgvals)
            plt.scatter(Y_train_standarized.squeeze()[:, 0], Y_train_standarized.squeeze()[:, 1], color="orange")
            plt.scatter(mean_best[0], mean_best[1], color="red")
            plt.show()



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

        if self.num_objectives in con_indices:
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

    @staticmethod
    def kgcb(a: Tensor, b: Tensor) -> Tensor:
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
        assert len(a) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"


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
        maxa = torch.max(a)
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

        if kg<-1e-3:
            print("kg",kg)
            print("kavals", torch.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:])))
            print("maxa",maxa)
            print("a",a)
            print("b", b)
            raise
        return kg

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.X_discretisation_size, :]



def construct_z_vals(nz: int, device: Optional[torch.device] = None) -> Tensor:
    """make nz equally quantile-spaced z values"""

    quantiles_z = torch.linspace(start=0.001, end=0.999, steps=nz)
    normal = torch.distributions.Normal(0, 1)
    z_vals = normal.icdf(quantiles_z)
    return z_vals.to(device=device)


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
