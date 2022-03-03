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

from .baseoptimizer import BaseOptimizer
from .utils import timeit

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
        # This optimizer uses "L-BFGS-B" by default. If specified, optimizer is Adam.
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

            X_random_initial_conditions_raw = torch.rand((self.optional["RAW_SAMPLES"], self.dim))
            x_GP_rec = self.policy()
            X_sampled = self.x_train

            # print(x_GP_rec.shape, X_random_initial_conditions_raw.shape, X_sampled.shape)
            X_initial_conditions_raw = torch.concat([X_random_initial_conditions_raw, x_GP_rec, X_sampled])
            X_initial_conditions_raw = X_initial_conditions_raw.unsqueeze(dim=-2)
            with torch.no_grad():
                mu_val_initial_conditions_raw = acq_fun.forward(X=X_initial_conditions_raw)

            best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[:self.optional["NUM_RESTARTS"]]
            X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]


            X_optimised, X_optimised_vals = gen_candidates_scipy(
                acquisition_function=acq_fun,
                initial_conditions=X_initial_conditions,
                lower_bounds=torch.zeros(self.dim),
                upper_bounds=torch.ones(self.dim),
            )

            x_best = X_optimised[torch.argmax(X_optimised_vals)]

        return x_best
