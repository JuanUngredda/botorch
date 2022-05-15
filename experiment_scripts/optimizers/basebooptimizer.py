import logging
import sys
from typing import Optional

import torch
from botorch.generation import gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from torch import Tensor
from botorch.generation.gen import gen_candidates_scipy
from .baseoptimizer import BaseOptimizer
from .utils import timeit, generate_unconstrained_best_samples

LOG_FORMAT = (
    "%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s:  %(message)s"
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class BaseBOOptimizer(BaseOptimizer):
    def __init__(
        self,
        testfun,
        acquisitionfun,
        lb: Tensor,
        ub: Tensor,
        n_max: int,
        n_init: int = 20,
        optional: Optional[dict[str, int]] = None,
    ):
        """
        kernel_str: string, SE or Matern
        n_ms: int, number of multi starts in Adam
        adam_iters: number of iterations for each atam ruin
        """

        self.acquisition_fun = acquisitionfun
        super().__init__(testfun, lb, ub, n_max, n_init, ns0=n_init)

        if optional is None:
            self.optional = {
                "OPTIMIZER": "Default",
                "NOISE_OBJECTIVE": None,
                "RAW_SAMPLES": 80,
                "NUM_RESTARTS": 5,
            }
        else:
            if optional["RAW_SAMPLES"] is None:
                optional["RAW_SAMPLES"] = 80

            if optional["NUM_RESTARTS"] is None:
                optional["NUM_RESTARTS"] = 5

            if optional["OPTIMIZER"] is None:
                optional["OPTIMIZER"] = "Default"

            if optional["NOISE_OBJECTIVE"] is False:
                optional["NOISE_OBJECTIVE"] = False

            self.optional = optional

    @timeit
    def _sgd_optimize_aqc_fun(self, acq_fun: callable, **kwargs) -> Tensor:
        """Use multi-start Adam SGD over multiple seeds"""

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])
        import time
        time_s = time.time()
        with torch.no_grad():

            X_initial_conditions_raw, weights, _ = acq_fun._initialize_maKG_parameters(model=self.model)


            X_initial_conditions_best_uncons_post_mean = generate_unconstrained_best_samples(model=self.model,
                                                                                             utility_model=self.utility_model,
                                                                                             weights=weights,
                                                                                             input_dim=self.dim,
                                                                                             optional=self.optional,
                                                                                             bounds=bounds_normalized,
                                                                                             num_objectives=self.y_train.shape[-1])
            X_initial_conditions_raw = torch.cat([X_initial_conditions_raw, X_initial_conditions_best_uncons_post_mean])
            # print("X_initial_conditions_raw",X_initial_conditions_raw, X_initial_conditions_raw.shape)
            X_initial_condition_random = torch.rand((10, 1, self.dim))
            X_initial_conditions_raw = torch.cat([X_initial_conditions_raw, X_initial_condition_random])
            # print("X_initial_conditions_raw ",X_initial_conditions_raw ,X_initial_conditions_raw.shape)
            # raise
            mu_val_initial_conditions_raw = acq_fun.forward(X_initial_conditions_raw)

            best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                             : self.optional["NUM_RESTARTS"]
                             ].squeeze()

            X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]
        time_e = time.time()

        # This optimizer uses "L-BFGS-B" by default. If specified, optimizer is Adam.
        if self.optional["OPTIMIZER"] == "Adam":
            x_best, _ = gen_candidates_torch(
                initial_conditions=X_initial_conditions.unsqueeze(dim=-2),
                acquisition_function=acq_fun,
                lower_bounds=bounds_normalized[0, :],
                upper_bounds=bounds_normalized[1, :],
                optimizer=torch.optim.Adam,
            )
        else:
            x_best, _ = gen_candidates_scipy(
                acquisition_function=acq_fun,
                initial_conditions=X_initial_conditions.unsqueeze(dim=-2),
                lower_bounds=torch.zeros(self.dim),
                upper_bounds=torch.ones(self.dim)
            )

        time_e = time.time()

        # print("X_initial_conditions", X_initial_conditions, "x_best",x_best, "_")
        # posterior_best = self.model.posterior(x_best)
        # mean_best = posterior_best.mean.squeeze().detach().numpy()
        # print("mean_best",mean_best, "_", _)
        # raise
        # with torch.no_grad():
        #     self.plot_points_on_objective(points=torch.atleast_2d(x_best), init_points=torch.atleast_2d(X_initial_conditions_raw))

        print("x_best", x_best, "value", _)
        return x_best.squeeze(dim=-2).detach()

    def plot_points_on_objective(self, points, init_points):
        plot_X = torch.rand((1000, 1, self.f.dim))
        init_points = init_points.squeeze()

        import matplotlib.pyplot as plt
        from botorch.utils.transforms import unnormalize

        Y_train_standarized = self.y_train

        with torch.no_grad():
            bounds = torch.vstack([self.lb, self.ub])
            x = unnormalize(X=plot_X, bounds=bounds)
            objective_best_vals = torch.vstack([self.f(x_i) for x_i in points]).to(dtype=torch.double)
            init_points_vals = torch.vstack([self.f(x_i) for x_i in init_points]).to(dtype=torch.double)

            objective = torch.vstack([self.f(x_i) for x_i in x]).to(dtype=torch.double)
            constraints = -torch.vstack([self.f.evaluate_slack(x_i) for x_i in x]).to(dtype=torch.double)
            is_feas = (constraints.squeeze() <= 0)
            if len(is_feas.shape) == 1:
                is_feas = is_feas.unsqueeze(dim=-1)
            aggregated_is_feas = torch.prod(is_feas, dim=-1, dtype=bool)
            plt.scatter(objective[:, 0], objective[:, 1], color="grey")
            plt.scatter(objective[aggregated_is_feas, 0], objective[aggregated_is_feas, 1], color="green")
            plt.scatter(Y_train_standarized.squeeze()[:,0],Y_train_standarized.squeeze()[:,1], color="orange", marker="x")
            plt.scatter(objective_best_vals[:,0], objective_best_vals[:,1], color="red")
            plt.scatter(init_points_vals[:,0], init_points_vals[:,1], color="magenta", marker="x")
            plt.show()
