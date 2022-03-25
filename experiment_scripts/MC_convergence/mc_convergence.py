import os
import pickle as pkl
from typing import Optional

import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils import standardize
from botorch.utils.transforms import unnormalize, normalize
from .basebooptimizer import BaseBOOptimizer
from .utils import timeit
from gpytorch.kernels.matern_kernel import MaternKernel
from .test_functions.gp_synthetic_test_function import GP_synthetic

class Optimizer(BaseBOOptimizer):
    def __init__(
            self,
            testfun,
            acquisitionfun,
            lb,
            ub,
            n_max: int,
            n_init: int = 20,
            kernel_str: str = None,
            nz: int = 5,
            base_seed: Optional[int] = 0,
            save_folder: Optional[str] = None,
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
        if kernel_str == "RBF":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim),
            )
        elif kernel_str == "Matern":
            self.covar_module = ScaleKernel(
                MaternKernel(ard_num_dims=self.dim),
            )
        else:
            raise Exception("Expected RBF or Matern Kernel")


    @timeit
    def evaluate_objective(self, x: Tensor, **kwargs) -> Tensor:

        # bring x \in [0,1]^d to original bounds.
        x = unnormalize(X=x, bounds=self.bounds)
        y = torch.Tensor([self.f(x)]).to(dtype=torch.double)
        return y

    def _update_model(self, X_train: Tensor, Y_train: Tensor):

        # Standarize traint Y values to Normal(0,1).
        Y_train_standarized = standardize(Y_train)


        if self.f.problem == "GP_synthetic":
            self.covar_module = self.f.covar_module

            # We can specify that it's deterministic and adding some small noise for numerical stability.
            NOISE_VAR = torch.Tensor([1e-4])

            self.model = FixedNoiseGP(
                train_X=X_train,
                train_Y=Y_train_standarized,
                covar_module=self.covar_module,
                train_Yvar=NOISE_VAR.expand_as(Y_train_standarized),
            )

        else:
            if self.optional["NOISE_OBJECTIVE"]:
                # We can specify that it's noisy and learn the noise by maximum likelihood.
                self.model = SingleTaskGP(
                    train_X=X_train,
                    train_Y=Y_train_standarized,
                    covar_module=self.covar_module,
                )
            else:
                # We can specify that it's deterministic and adding some small noise for numerical stability.
                NOISE_VAR = torch.Tensor([1e-4])

                self.model = FixedNoiseGP(
                    train_X=X_train,
                    train_Y=Y_train_standarized,
                    covar_module=self.covar_module,
                    train_Yvar=NOISE_VAR.expand_as(Y_train_standarized),
                )
                raise
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_model(mll)


    def policy(self):

        self._update_model(self.x_train, self.y_train)
        x_rec, x_rec_val = self.best_model_posterior_mean(model=self.model)

        return x_rec, x_rec_val

    def best_model_posterior_mean(self, model):
        """find the highest predicted x to return to the user"""

        assert self.y_train is not None
        "Include data to find best posterior mean"

        bounds_normalized = torch.vstack(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
        )

        # generate initialisation points
        X_random_initial_conditions_raw = torch.rand((self.optional["RAW_SAMPLES"], self.dim))
        X_sampled = self.x_train

        # print(x_GP_rec.shape, X_random_initial_conditions_raw.shape, X_sampled.shape)
        X_initial_conditions_raw = torch.concat([X_random_initial_conditions_raw, X_sampled])
        X_initial_conditions_raw = X_initial_conditions_raw.unsqueeze(dim=-2)

        with torch.no_grad():
            x_train_posterior_mean = PosteriorMean(model).forward(X_initial_conditions_raw).squeeze()

        best_k_indeces = torch.argsort(x_train_posterior_mean, descending=True)[:self.optional["NUM_RESTARTS"]]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]


        X_optimised, X_optimised_vals = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=bounds_normalized,
            batch_initial_conditions=X_initial_conditions,
            q=1,
            num_restarts=self.optional["NUM_RESTARTS"],
            raw_samples=self.optional["RAW_SAMPLES"],
        )

        x_best = X_optimised[torch.argmax(X_optimised_vals.squeeze())]

        return torch.atleast_2d(x_best), torch.max(X_optimised_vals.squeeze())

    def get_next_point(self):
        self._update_model(self.x_train, self.y_train)

        x_GP_rec, x_GP_rec_val = self.policy()
        acquisition_function = self.acquisition_fun(self.model, x_optimiser= x_GP_rec, current_value=x_GP_rec_val)
        print(acquisition_function)
        x_new, _ = self._sgd_optimize_aqc_fun(
            acquisition_function, log_time=self.method_time, log_acq_vals= self.acq_vals)

        return x_new

    def save(self):
        # save the output
        ynoise = torch.unique(self.model.likelihood.noise_covar.noise)
        gp_likelihood_noise = torch.Tensor([ynoise])
        gp_lengthscales = self.model.covar_module.base_kernel.lengthscale.detach()
        self.gp_likelihood_noise = torch.cat(
            [self.gp_likelihood_noise, gp_likelihood_noise]
        )
        self.gp_lengthscales = torch.cat([self.gp_lengthscales, gp_lengthscales])
        self.kernel_name = str(self.model.covar_module.base_kernel.__class__.__name__)
        self.optimal_value = self.f.optimal_value


        output = {
            "problem": self.f.problem,
            "method_times": self.method_time,
            "acq_outputs": self.acq_vals,
            "OC": self.performance,
            "optimum": self.optimal_value,
            "x": unnormalize(self.x_train, self.bounds),
            "evaluation_time": self.evaluation_time,
            "y": self.y_train,
            "kernel": self.kernel_name,
            "gp_lik_noise": self.gp_likelihood_noise,
            "gp_lengthscales": self.gp_lengthscales,
            "base_seed": self.base_seed,
            "cwd": os.getcwd(),
            "savefile": self.save_folder,
        }

        if self.save_folder is not None:
            if os.path.isdir(self.save_folder) == False:
                os.makedirs(self.save_folder)

            with open(self.save_folder + "/" + str(self.base_seed) + ".pkl", "wb") as f:
                pkl.dump(output, f)
