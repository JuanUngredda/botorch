from typing import List, Tuple, Optional

import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from torch import Tensor

from botorch.test_functions.base import BaseTestProblem
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.initializers import gen_value_function_initial_conditions
from scipy.optimize import minimize

dtype = torch.double

class SyntheticTestFunction(BaseTestProblem):
    r"""Base class for synthetic test functions."""

    _optimizers: List[Tuple[float, ...]]
    _optimal_value: float
    num_objectives: int = 1

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Base constructor for synthetic test functions.

        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    def save_optimal_value(self):
        if self._optimizers is not None:
            self.register_buffer(
                "optimizers", torch.tensor(self._optimizers, dtype=torch.float)
            )

    @property
    def optimal_value(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        return -self._optimal_value if self.negate else self._optimal_value


class GP_synthetic(SyntheticTestFunction):
    """
A toy function GP

ARGS
 min: scalar defining min range of inputs
 max: scalar defining max range of inputs
 seed: int, RNG seed
 x_dim: designs dimension
 a_dim: input dimensions
 xa: n*d matrix, points in space to eval testfun
 NoiseSD: additive gaussaint noise SD

RETURNS
 output: vector of length nrow(xa)
 """

    def __init__(self, noise_std: Optional[float] = None,
                 negate: Optional[bool] = False,
                 kernel_str: Optional[str] = "Matern",
                 hypers_ls: Optional[float] = 0.5,
                 seed: Optional[int] = 1,
                 dim: Optional[int] = 1) -> None:

        self.seed = seed
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]

        hypers = {
            'covar_module.base_kernel.lengthscale': torch.tensor(hypers_ls),
            'covar_module.outputscale': torch.tensor(1.),
        }

        super().__init__(noise_std=noise_std, negate=negate)

        if kernel_str == "RBF":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim),
            )

        elif kernel_str == "Matern":
            self.covar_module = ScaleKernel(
                MaternKernel(ard_num_dims=self.dim),
            )

        l = hypers['covar_module.base_kernel.lengthscale']
        vr = hypers['covar_module.outputscale']


        self.covar_module.base_kernel.lengthscale = l
        self.covar_module.outputscale = vr

        self.generate_function()
        optimizers, optvals = self.optimize_optimal_value()
        self._optimal_value = optvals
        self._optimizers = optimizers
        self.save_optimal_value()


    def optimize_optimal_value(self):
        r"""The global minimum (maximum if negate=True) of the function."""

        bounds = torch.vstack(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
        )

        X_initial_conditions_raw = torch.rand((100000,  self.dim)).to(dtype=dtype)
        X_initial_conditions_raw = torch.concat([X_initial_conditions_raw, bounds])
        mu_val_initial_conditions_raw = self.forward(X_initial_conditions_raw).squeeze()
        best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[:5]

        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

        self.best_val = 9999
        def wrapped_evaluate_true_fun(X):
            X = torch.atleast_2d(torch.Tensor(X).to(dtype=dtype))
            ub_condition = X <= bounds[1] + 1e-4
            lb_condition = X >= bounds[0] - 1e-4
            overall_condition = torch.prod(ub_condition * lb_condition, dtype=bool)
            if overall_condition:
                val = -self.forward(X).to(dtype=dtype).squeeze().detach().numpy()
                if val < self.best_val:
                    self.best_val = val
                return val
            else:
                return (torch.ones((X.shape[0]))*999).numpy()


        res = [minimize(wrapped_evaluate_true_fun, x0=x0, method='nelder-mead', tol=1e-9) for x0 in X_initial_conditions]
        x_best = res[0]["x"]
        # x_best_val = res["fun"]

        return list(x_best), self.best_val

    def evaluate_true(self, X: Tensor) -> Tensor:

        ks = self.covar_module.forward(X, self.x_base_points)
        out = torch.matmul(ks, self.invCZ)
        return out

    def generate_function(self):
        print("Generating test function")
        torch.manual_seed(self.seed)

        self.x_base_points = torch.rand(size=(50, self.dim))

        mu = torch.zeros((1,self.x_base_points.shape[0])).to(dtype=dtype)

        C = self.covar_module.forward(self.x_base_points, self.x_base_points).to(dtype=dtype)
        C+= torch.eye(C.shape[0]) * 1e-3
        mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=C)
        Z = mvn.rsample(sample_shape=(1, len(mu))).view( C.shape[0], 1)

        invC = torch.linalg.inv(C )

        self.invCZ = torch.matmul(invC, Z)


# GP_TEST = GP_synthetic(negate=True,  hypers_ls=0.8, seed=2, dim=2).to(dtype=dtype)
# #
# X_plot = torch.rand((10000, 2))
# #
# fval = GP_TEST(X_plot).detach()
# optimal_xstar = GP_TEST._optimizers
# print("xstar",optimal_xstar)
# import matplotlib.pyplot as plt
# plt.scatter(X_plot[:,0], X_plot[:,1], c=fval)
# plt.scatter(optimal_xstar[0], optimal_xstar[1], color="red")
# plt.show()
# # plt.scatter(X_plot, fval)
# # plt.show()