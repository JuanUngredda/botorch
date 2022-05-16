import logging
import sys
from typing import Optional, Tuple

import torch
from botorch.generation import gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from torch import Tensor
from botorch.generation.gen import gen_candidates_scipy
from .baseoptimizer import BaseOptimizer
from .utils import timeit
from botorch.acquisition.multi_objective.multi_attribute_constrained_kg import MultiAttributeConstrainedKG, MultiAttributePenalizedKG
from botorch.utils.sampling import sample_simplex

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


    @timeit
    def _sgd_optimize_aqc_fun(self, acq_fun: callable,
                              bacth_initial_points: Optional[Tensor]=None, **kwargs) -> Tensor:
        """Use multi-start Adam SGD over multiple seeds"""

        if isinstance(acq_fun, (MultiAttributeConstrainedKG, MultiAttributePenalizedKG)):
            bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])
            import time
            num_xnew = self.optional["RAW_SAMPLES"]
            num_xstar = acq_fun.X_discretisation_size

            ts = time.time()

            xnew_samples = bacth_initial_points
            print("xnew_samples", xnew_samples.shape)

            num_rd_samples = 0
            if num_xnew > bacth_initial_points.shape[0]:
                num_rd_samples = num_xnew - bacth_initial_points.shape[0]
                xnew_samples_rd = torch.rand((num_rd_samples , 1, self.dim))
                xnew_samples = torch.vstack([xnew_samples, xnew_samples_rd])

            te = time.time()
            print("gen xnew", ts-te)

            X_initial_conditions = torch.zeros((num_xnew + num_rd_samples, num_xstar + 1, self.dim), dtype=torch.double)

            ts = time.time()
            xstar = torch.rand((num_xstar , 1, self.dim))
            xstar = xstar.squeeze(dim=-2)
            te = time.time()
            print("gen xnew", ts-te)

            for xnew_idx, xnew in enumerate(xnew_samples):
                idx_ics = torch.cat([xnew, xstar]).unsqueeze(dim=0)
                X_initial_conditions[xnew_idx, ...] = idx_ics

            # print(batch_initial_conditions)
            ts = time.time()
            with torch.no_grad():
                mu_val_initial_conditions_raw = acq_fun.forward(X_initial_conditions)

                best_k_indeces = torch.argsort(mu_val_initial_conditions_raw.squeeze(), descending=True)[
                                 : self.optional["NUM_RESTARTS"]
                                 ].squeeze()

                batch_initial_conditions = X_initial_conditions[best_k_indeces:best_k_indeces+1, :, :]
            te = time.time()
            # self.plot_points_on_objective(points=xnew_samples.squeeze(),
            #                               cval=mu_val_initial_conditions_raw,
            #                               scalarizations=xnew_weights)
            # print("batch_initial_conditions ",batch_initial_conditions, te-ts )
            # acq_fun._plot(X=batch_initial_conditions,
            #               lb = self.lb,
            #               ub = self.ub,
            #               X_train=self.x_train,
            #               Y_train=self.y_train,
            #               C_train=self.c_train,
            #               true_fun=self.f)
            # print("optimising acq")
            import time
            ts = time.time()
            x_best_concat, _ = optimize_acqf(
                acq_function=acq_fun,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.optional["NUM_RESTARTS"],
                batch_initial_conditions=batch_initial_conditions,
                optimizer=torch.optim.Adam,
                return_full_tree=False
            )
            te = time.time()
            print("opt time", te-ts)
            # print("finished optimising acq")
            x_best = acq_fun.extract_candidates(X_full=x_best_concat)
            # print("xbest", x_best)
            # with torch.no_grad():
            #     self.plot_points_on_objective(points=torch.atleast_2d(x_best),
            #                                   init_points=torch.atleast_2d(X_initial_conditions))
            # print("plot on x_best_concat")
            # acq_fun._plot(X=x_best_concat,
            #               lb = self.lb,
            #               ub = self.ub,
            #               X_train=self.x_train,
            #               Y_train=self.y_train,
            #               C_train=self.c_train,
            #               true_fun=self.f)
        else:
            bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])
            x_best, _ = optimize_acqf(
                acq_function=acq_fun,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.optional["NUM_RESTARTS"],
                raw_samples=self.optional["RAW_SAMPLES"], # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
        return x_best


