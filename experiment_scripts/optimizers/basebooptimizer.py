import logging
import sys
from typing import Optional

import torch
from botorch.generation import gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from torch import Tensor
from botorch.generation.gen import gen_candidates_scipy
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.knowledge_gradient import (qKnowledgeGradient, HybridOneShotKnowledgeGradient)
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
    gen_hybrid_one_shot_kg_initial_conditions,
)

from .baseoptimizer import BaseOptimizer
from .utils import timeit, RandomSample

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

        bounds_normalized = torch.vstack(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
        )

        if isinstance(acq_fun, RandomSample):
            x_best = torch.rand((1, self.dim))
            return x_best

        if isinstance(acq_fun, HybridOneShotKnowledgeGradient):
            batch_initial_conditions = gen_hybrid_one_shot_kg_initial_conditions(
                acq_function=acq_fun,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.optional["NUM_RESTARTS"],
                raw_samples=self.optional["RAW_SAMPLES"]
            )

            x_best, _ = optimize_acqf(
                acq_function=acq_fun,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.optional["NUM_RESTARTS"],
                batch_initial_conditions=batch_initial_conditions,
                return_full_tree=False
            )

            return x_best
        # batch_initial_conditions = gen_hybrid_one_shot_kg_initial_conditions(
        #     acq_function=acq_fun,
        #     bounds=bounds_normalized,
        #     q=1,
        #     num_restarts=self.optional["NUM_RESTARTS"],
        #     raw_samples=self.optional["RAW_SAMPLES"]
        # )
        #
        # _ = acq_fun.evaluate_discrete_kg(X=batch_initial_conditions, test=True)
        #
        #
        # x_best, _ = optimize_acqf(
        #     acq_function=acq_fun,
        #     bounds=bounds_normalized,
        #     q=1,
        #     num_restarts=self.optional["NUM_RESTARTS"],
        #     batch_initial_conditions = batch_initial_conditions,
        #     return_full_tree= True
        # )
        #
        # _ = acq_fun.evaluate_discrete_kg(X=x_best, test=True)
        #
        # from botorch.utils.sampling import (
        #     draw_sobol_samples
        # )
        # import matplotlib.pyplot as plt
        # from botorch.acquisition.analytic import DiscreteKnowledgeGradient
        #
        # with torch.no_grad():
        #     X_discretisation = draw_sobol_samples(
        #         bounds=bounds_normalized, n=100, q=1
        #     )
        #
        #     X_plot = draw_sobol_samples(
        #         bounds=bounds_normalized, n=1000, q=1
        #     )
        #
        #     kgvals = torch.zeros(X_plot.shape[0], dtype=torch.double)
        #     optimal_kgvals = torch.zeros(X_plot.shape[0], dtype=torch.double)
        #
        #     for x_i, xnew in enumerate(X_plot):
        #         xnew = torch.atleast_2d(xnew.squeeze())
        #         optimal_discretisation = x_best[1:,:].squeeze()
        #
        #         optimal_kgvals[x_i] = DiscreteKnowledgeGradient.compute_discrete_kg(model=self.model,
        #                                                                 xnew=xnew,
        #                                                                optimal_discretisation=optimal_discretisation)
        #
        #     for x_i, xnew in enumerate(X_plot):
        #         xnew = torch.atleast_2d(xnew.squeeze())
        #         X_discretisation = X_discretisation.squeeze()
        #         kgvals[x_i] = DiscreteKnowledgeGradient.compute_discrete_kg(model=self.model,
        #                                                                 xnew=xnew,
        #                                                                optimal_discretisation=X_discretisation)
        #
        #
        #     print("max min",torch.max(kgvals), torch.min(kgvals))
        #
        # print(x_best.shape)
        # print(batch_initial_conditions.shape)
        # # raise
        # X_plot =X_plot.squeeze().numpy()
        # plt.scatter(X_plot[:,0], X_plot[:,1] , c = kgvals.detach().numpy())
        # plt.scatter(x_best[:,0], x_best[:,1], color="red")
        # plt.scatter(x_best[0, 0], x_best[0, 1], color="red", marker="x", s=100)
        # plt.scatter(batch_initial_conditions[0,0,0], batch_initial_conditions[0,0,1], color="magenta", marker="x", s=100)
        # plt.scatter(batch_initial_conditions[0, :, 0], batch_initial_conditions[0, :, 1], color="magenta")
        #
        # # x_GP_rec = self.policy()
        # # plt.scatter(x_GP_rec[:,0], x_GP_rec[:,1], color="magenta")
        # plt.show()
        # # raise
        # # X_plot =X_plot.squeeze()
        # # plt.scatter(X_plot[:,0], X_plot[:,1] , c = optimal_kgvals.detach())
        # # plt.scatter(x_best[:,0], x_best[:,1], color="red")
        # # plt.scatter(batch_initial_conditions[0,0,0], batch_initial_conditions[0,0,1], color="red", marker="x")
        # #
        # #
        # # # plt.scatter(x_GP_rec[:,0], x_GP_rec[:,1], color="magenta")
        # # plt.show()


        # This optimizer uses "L-BFGS-B" by default. If specified, optimizer is Adam.
        if isinstance(acq_fun, qKnowledgeGradient):
            x_best, _ = optimize_acqf(
                acq_function=acq_fun,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.optional["NUM_RESTARTS"],
                raw_samples=self.optional["RAW_SAMPLES"]
            )

            return x_best


        if self.optional["OPTIMIZER"] == "Adam":
            initial_conditions = gen_batch_initial_conditions(
                acq_function=acq_fun,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.optional["NUM_RESTARTS"],
                raw_samples=self.optional["RAW_SAMPLES"],
            )
            x_best, _ = gen_candidates_torch(
                initial_conditions=initial_conditions,
                acquisition_function=acq_fun,
                lower_bounds=bounds_normalized[0, :],
                upper_bounds=bounds_normalized[1, :],
                optimizer=torch.optim.Adam,
            )
        else:
            import time
            ts = time.time()
            X_random_initial_conditions_raw = torch.rand((self.optional["RAW_SAMPLES"], self.dim))

            x_GP_rec = self.policy()
            X_sampled = self.x_train
            Y_train = self.y_train
            X_best_sampled = torch.atleast_2d(X_sampled[torch.argmax(Y_train)])
            # print(x_GP_rec.shape, X_random_initial_conditions_raw.shape, X_sampled.shape)
            X_initial_conditions_raw = torch.concat([X_random_initial_conditions_raw, x_GP_rec, X_best_sampled ])
            X_initial_conditions_raw = X_initial_conditions_raw.unsqueeze(dim=-2)

            with torch.no_grad():
                mu_val_initial_conditions_raw = acq_fun.forward(X=X_initial_conditions_raw).squeeze()

            best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[:self.optional["NUM_RESTARTS"]]
            X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

            # X_optimised_torch, X_optimised_vals = gen_candidates_torch(
            #     acquisition_function=acq_fun,
            #     initial_conditions=X_initial_conditions,
            #     lower_bounds=torch.zeros(self.dim),
            #     upper_bounds=torch.ones(self.dim),
            #     optimizer=torch.optim.Adam,
            # )
            #
            # xval_torch = X_optimised_torch[torch.argmax(X_optimised_vals.squeeze())].squeeze().detach().numpy()
            te = time.time()
            print("time init", te-ts)
            X_optimised, X_optimised_vals = gen_candidates_scipy(
                acquisition_function=acq_fun,
                initial_conditions=X_initial_conditions,
                lower_bounds=torch.zeros(self.dim),
                upper_bounds=torch.ones(self.dim),
            )
            te = time.time()
            print("time", te-ts)
            raise
            # xval = X_optimised[torch.argmax(X_optimised_vals.squeeze())].squeeze().detach().numpy()

            # print("X_initial_conditions",X_initial_conditions)
            # print("X_optimised",  X_optimised_torch, X_optimised_scipy)
            # internal check. eliminate after
            # X_unit_cube_samples = torch.rand((100, 1,  self.dim))
            # X_initial_conditions_raw = X_unit_cube_samples
            # #
            # X_initial_conditions_raw = torch.concat([X_initial_conditions_raw,
            #                                          X_optimised_torch,
            #                                          X_optimised_scipy])
            # with torch.no_grad():
            #     mu_val_initial_conditions_raw = acq_fun.forward(X_initial_conditions_raw)
            #
            # Xplot = X_initial_conditions_raw.squeeze().detach().numpy()
            #
            # import matplotlib.pyplot as plt
            #
            # X_init = X_initial_conditions.squeeze().numpy()
            # plt.scatter(Xplot[:,0], Xplot[:,1], c=mu_val_initial_conditions_raw.numpy().squeeze())
            # plt.scatter(X_init[0], X_init[1], color="red")
            # plt.scatter(xval_torch[ 0], xval_torch[1], color="red", marker="x")
            # plt.scatter(xval_scipy[0], xval_scipy[1], color="blue", marker="x")
            # plt.show()
            #
            #
            # raise
            x_best = X_optimised[torch.argmax(X_optimised_vals.squeeze())]

        return x_best
