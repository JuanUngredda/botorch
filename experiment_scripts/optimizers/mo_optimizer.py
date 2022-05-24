import os
import pickle as pkl
from typing import Optional

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.scalarization import (
    get_chebyshev_scalarization,
    get_linear_scalarization,
)
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from botorch.utils import standardize
from botorch.utils.sampling import (
    draw_sobol_samples)

from .basebooptimizer import BaseBOOptimizer
from .utils import timeit, UnconstrainedParetoFrontApproximation, ParetoFrontApproximation, \
    TrueParetoFrontApproximation, _compute_expected_utility, ParetoFrontApproximation_xstar


class Optimizer(BaseBOOptimizer):
    def __init__(
            self,
            testfun,
            acquisitionfun,
            lb,
            ub,
            utility_model_name: str,
            num_scalarizations: int,
            n_max: int,
            n_init: int = 20,
            kernel_str: str = None,
            nz: int = 5,
            base_seed: Optional[int] = 0,
            save_folder: Optional[str] = None,
            is_noise: Optional[bool] =False,
            optional: Optional[dict[str, int]] = None,
    ):

        super().__init__(
            testfun,
            acquisitionfun,
            lb,
            ub,
            n_max=n_max,
            n_init=n_init,
            optional=optional,
        )

        torch.manual_seed(base_seed)
        self.base_seed = base_seed
        self.nz = nz
        self.save_folder = save_folder
        self.bounds = testfun.bounds
        self.num_scalarisations = num_scalarizations
        self.is_noise = is_noise
        if kernel_str == "RBF":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim),
            )
        elif kernel_str == "Matern":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim),
            )
        else:
            raise Exception("Expected RBF or Matern Kernel")

        if utility_model_name == "Tche":
            self.utility_model = get_chebyshev_scalarization

        elif utility_model_name == "Lin":
            self.utility_model = get_linear_scalarization

        self.weights = sample_simplex(
            n=self.num_scalarisations, d=self.f.num_objectives,
            qmc=True).squeeze()

    @timeit
    def evaluate_objective(self, x: Tensor, noise: Optional[bool] = True, **kwargs) -> Tensor:
        x = torch.atleast_2d(x)
        x = unnormalize(X=x, bounds=self.bounds)
        objective = self.f(x, noise=noise)
        return objective

    def evaluate_constraints(self, x: Tensor, noise: Optional[bool] = True, **kwargs) -> Tensor:
        x = torch.atleast_2d(x)
        x = unnormalize(X=x, bounds=self.bounds)
        constraints = -self.f.evaluate_slack(x, noise=noise)
        return constraints

    def _update_multi_objective_model_prediction_with_noise(self):


        models = []
        for i in range(self.y_train.shape[-1]):
            models.append(
                SingleTaskGP(self.x_train, self.y_train[..., i: i + 1])
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        bounds = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        with torch.no_grad():
            X_discretisation = draw_sobol_samples(
                bounds=bounds, n=1000, q=1
            )

            pred = model.posterior(X_discretisation).mean.squeeze()

        return pred

    def _update_multi_objective_model_prediction_without_noise(self):

        NOISE_VAR = torch.Tensor([1e-4])
        while True:
            try:
                models = []
                for i in range(self.y_train.shape[-1]):
                    models.append(
                        FixedNoiseGP(self.x_train, self.y_train[..., i: i + 1],
                                     train_Yvar=NOISE_VAR.expand_as(self.y_train[..., i: i + 1])
                                     )
                    )
                model = ModelListGP(*models)
                mll = SumMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)

                break
            except:
                print("xstar: increased assumed fixed noise term")
                NOISE_VAR *= 10
                print("original noise var:", 1e-4, "updated noisevar:", NOISE_VAR)

        bounds = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        with torch.no_grad():
            X_discretisation = draw_sobol_samples(
                bounds=bounds, n=1000, q=1
            )

            pred = model.posterior(X_discretisation).mean.squeeze()

        return pred

    def _update_model(self):

        self.weights = sample_simplex(n=self.num_scalarisations, d=self.f.num_objectives, qmc=True).squeeze()
        self.weights = torch.atleast_2d(self.weights)

        if self.is_noise:
            pred = self._update_multi_objective_model_prediction_with_noise()
        else:
            pred = self._update_multi_objective_model_prediction_without_noise()
        self.pred = pred
        if self.is_noise:
            print("model for noisy data")
            self.model = self.train_scalarized_objectives_with_noise(normalizing_vectors=pred,
                                                                        weights=self.weights)
        else:
            print("model for deterministic data")
            self.model = self.train_scalarized_objectives_without_noise(normalizing_vectors=pred,
                                                                        weights=self.weights)

    def best_model_posterior_mean(self, weights):
        """find the highest predicted x to return to the user"""

        assert self.y_train is not None
        "Include data to find best posterior mean"

        X_pareto_solutions, _ = ParetoFrontApproximation(
            model=self.model,
            input_dim=self.dim,
            weights=weights,
            num_objectives=weights.shape[0],
            num_constraints=self.c_train.shape[-1],
            optional=self.optional,
        )

        return X_pareto_solutions, weights

    def best_unconstrained_model_posterior_mean(self, weights):
        """find the highest predicted x to return to the user"""

        assert self.y_train is not None
        "Include data to find best posterior mean"

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        if self.is_noise:
            model = self.train_scalarized_objectives_with_noise(normalizing_vectors=self.pred,
                                                                        weights=weights)
        else:
            model = self.train_scalarized_objectives_without_noise(normalizing_vectors=self.pred,
                                                                        weights=weights)


        X_pareto_solutions, _ = UnconstrainedParetoFrontApproximation(
            model=model,
            objective_dim=self.y_train.shape[1],
            scalatization_fun=self.utility_model,
            input_dim=self.dim,
            bounds=bounds_normalized,
            y_train=self.y_train,
            x_train=self.x_train,
            c_train=self.c_train,
            weights=weights,
            num_objectives=weights.shape[0],
            num_constraints=self.c_train.shape[-1],
            optional=self.optional,
        )

        return X_pareto_solutions, weights

    def train_scalarized_objectives_without_noise(self, normalizing_vectors, weights):

        NOISE_VAR = torch.Tensor([1e-4])
        while True:
            try:
                models = []
                for w in weights:
                    scalarization_fun = self.utility_model(weights=w, Y=normalizing_vectors)
                    utility_values = scalarization_fun(self.y_train).unsqueeze(dim=-2).view(self.x_train.shape[0], 1)
                    utility_values = standardize(utility_values)

                    models.append(
                        FixedNoiseGP(self.x_train, utility_values,
                                     train_Yvar=NOISE_VAR.expand_as(utility_values)
                                     )
                    )

                for i in range(self.c_train.shape[-1]):
                    models.append(
                        FixedNoiseGP(self.x_train, self.c_train[..., i: i + 1],
                                     train_Yvar=NOISE_VAR.expand_as(self.c_train[..., i: i + 1]), )
                    )

                model = ModelListGP(*models)

                mll = SumMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)
                break
            except:
                print("xstar: increased assumed fixed noise term")
                NOISE_VAR *= 10
                print("original noise var:", 1e-4, "updated noisevar:", NOISE_VAR)

        return model

    def train_scalarized_objectives_with_noise(self, normalizing_vectors, weights):


        models = []
        for w in weights:
            scalarization_fun = self.utility_model(weights=w, Y=normalizing_vectors)
            utility_values = scalarization_fun(self.y_train).unsqueeze(dim=-2).view(self.x_train.shape[0], 1)
            utility_values = standardize(utility_values)

            models.append(SingleTaskGP(self.x_train, utility_values))

        for i in range(self.c_train.shape[-1]):
            models.append(
                SingleTaskGP(self.x_train, self.c_train[..., i: i + 1] ))

        model = ModelListGP(*models)

        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        return model


    def gen_xstar_values(self, model, weights):
        """find the highest predicted x to return to the user"""

        assert self.y_train is not None
        "Include data to find best posterior mean"

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        if self.is_noise:
            model_xstar = self.train_scalarized_objectives_with_noise(normalizing_vectors=self.pred,
                                                                        weights=weights)
        else:
            model_xstar = self.train_scalarized_objectives_without_noise(normalizing_vectors=self.pred,
                                                                        weights=weights)


        # print("weights", weights.shape)
        X_pareto_solutions, _ = ParetoFrontApproximation_xstar(
            model=model_xstar,
            objective_dim=self.y_train.shape[1],
            scalatization_fun=self.utility_model,
            input_dim=self.dim,
            bounds=bounds_normalized,
            y_train=self.y_train,
            x_train=self.x_train,
            c_train=self.c_train,
            weights=weights,
            num_objectives=weights.shape[0],
            num_constraints=self.c_train.shape[-1],
            optional=self.optional,
        )

        return X_pareto_solutions, weights

    def get_next_point(self):
        self._update_model()

        X_constrained_initial_conditions_raw, _ = self.best_model_posterior_mean(weights=self.weights)

        X_unconstrained_initial_conditions_raw, _ = self.best_unconstrained_model_posterior_mean(weights=self.weights)

        X_initial_conditions_raw = torch.cat([X_constrained_initial_conditions_raw,
                                              X_unconstrained_initial_conditions_raw])

        acquisition_function = self.acquisition_fun(self.model,
                                                    train_x=self.x_train,
                                                    train_obj=self.y_train,
                                                    train_con=self.c_train,
                                                    fixed_scalarizations=self.weights,
                                                    current_global_optimiser=X_initial_conditions_raw,
                                                    X_pending=None)
        x_new = self._sgd_optimize_aqc_fun(
            acquisition_function,
            bacth_initial_points=X_initial_conditions_raw,
            log_time=self.method_time)
        return x_new


    def policy(self):

        # print(self.x_train, self.y_train, self.c_train)

        self._update_model()

        x_rec = self.best_model_posterior_mean(weights=self.weights)

        return x_rec

    def save(self):
        # save the output

        self.gp_likelihood_noise = [
            self.model.likelihood.likelihoods[n].noise_covar.noise
            for n in range(self.model.num_outputs)
        ]

        self.gp_lengthscales = [
            self.model.models[n].covar_module.base_kernel.lengthscale.detach()
            for n in range(self.model.num_outputs)
        ]

        self.kernel_name = str(
            self.model.models[0].covar_module.base_kernel.__class__.__name__
        )

        output = {
            "problem": self.f.problem,
            "method_times": self.method_time,
            "OC_GP": self.GP_performance,
            "OC_sampled": self.sampled_performance,
            "x": self.x_train,
            "y": self.y_train,
            "c": self.c_train,
            "x_pareto_recommended": self.pareto_set_recommended,
            "weights": self.weights,
            "kernel": self.kernel_name,
            "gp_lik_noise": self.gp_likelihood_noise,
            "gp_lengthscales": self.gp_lengthscales,
            "base_seed": self.base_seed,
            "cwd": os.getcwd(),
            "savefile": self.save_folder,
        }

        if self.save_folder is not None:
            if os.path.isdir(self.save_folder) == False:
                os.makedirs(self.save_folder, exist_ok=True)

            with open(self.save_folder + "/" + str(self.base_seed) + ".pkl", "wb") as f:
                pkl.dump(output, f)


    def true_underlying_policy(self, weights):

        assert self.y_train is not None
        "Include data to find best posterior mean"
        X_pareto_solutions, _ = TrueParetoFrontApproximation(
            output_true_function=self.evaluate_objective,
            constraint_true_function=self.evaluate_constraints,
            input_dim=self.dim,
            weights=weights,
            x_train=self.x_train,
            x_recommended=self.pareto_set_recommended,
            normalizing_vectors=self.pred,
            utility_model=self.utility_model,
            num_objectives=weights.shape[0],
            num_constraints=self.c_train.shape[-1],
            optional=self.optional,
        )

        return X_pareto_solutions, weights

    def _compute_OC(self,
                    true_underlying_PF,
                    weights,
                    recommended_solutions):

        def utility_wrapper(utility):
            def objective(X):
                objective_values = torch.vstack([self.evaluate_objective(x_i, noise=False) for x_i in X]).to(
                    dtype=torch.double)
                constraint_values = torch.vstack([self.evaluate_constraints(x_i, noise=False) for x_i in X]).to(
                    dtype=torch.double)
                uvals = utility(objective_values)

                is_feas = constraint_values <= 0
                aggregated_is_feas = torch.prod(is_feas, dim=-1, dtype=int)
                objective_val = uvals * aggregated_is_feas

                return objective_val

            return objective

        OC = []
        for idx, w in enumerate(weights):
            utility = self.utility_model(weights=w, Y=self.pred)
            objective = utility_wrapper(utility=utility)

            true_best_uval = torch.max(objective(true_underlying_PF))
            estimated_best_uval = torch.max(objective(recommended_solutions))
            oc = torch.max(true_best_uval - estimated_best_uval, 0)
            OC.append(oc)
        OC = torch.Tensor(OC)
        expected_utility = torch.mean(OC)
        return expected_utility

    def test(self):
        """
        test and saves performance measures
        """

        self.pareto_set_recommended, self.weights = self.policy()
        self.true_underlying_recommended, _ = self.true_underlying_policy(weights=self.weights)
        estimated_OC = self._compute_OC(true_underlying_PF=self.true_underlying_recommended,
                                        weights=self.weights,
                                        recommended_solutions=self.pareto_set_recommended)

        sampled_OC = self._compute_OC(true_underlying_PF=self.true_underlying_recommended,
                                      weights=self.weights,
                                      recommended_solutions=self.x_train)

        n = len(self.y_train) * 1.0
        self.GP_performance = torch.vstack(
            [self.GP_performance, torch.Tensor([n, estimated_OC])]
        )

        self.sampled_performance = torch.vstack(
            [self.sampled_performance, torch.Tensor([n, sampled_OC])]
        )
        self.save()
