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
from .utils import timeit, acq_values_recorder
from botorch.acquisition.multi_objective.multi_attribute_constrained_kg import MultiAttributeConstrainedKG

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

    @acq_values_recorder
    @timeit
    def _sgd_optimize_aqc_fun(self, acq_fun: callable,
                              bacth_initial_points: Optional[Tensor]=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Use multi-start Adam SGD over multiple seeds"""

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        if bacth_initial_points is None:
            X_initial_conditions_raw = self.best_model_posterior_mean(model=self.model, weights=self.weights)
        else:
            X_initial_conditions_raw = bacth_initial_points

        if isinstance(acq_fun, MultiAttributeConstrainedKG):

            num_xnew = self.optional["NUM_RESTARTS"]
            num_xstar = acq_fun.X_discretisation_size
            input_dim = self.dim

            perm = torch.randperm(X_initial_conditions_raw.shape[0])
            idx = perm[:num_xnew]
            xnew_samples = X_initial_conditions_raw[idx]

            batch_initial_conditions = torch.zeros((num_xnew, num_xstar + 1, input_dim), dtype=torch.double)
            for xnew_idx, xnew in enumerate(xnew_samples):

                perm = torch.randperm(X_initial_conditions_raw.shape[0])
                idx = perm[:num_xstar]
                xstar = X_initial_conditions_raw[idx].squeeze(dim=-2)

                idx_ics = torch.cat([xnew, xstar]).unsqueeze(dim=0)
                batch_initial_conditions[xnew_idx, ...] = idx_ics

            x_best_concat, _ = optimize_acqf(
                acq_function=acq_fun,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.optional["NUM_RESTARTS"],
                batch_initial_conditions=batch_initial_conditions,
                return_full_tree=False
            )
            x_best = acq_fun.extract_candidates(X_full=x_best_concat)

            return x_best, _

        with torch.no_grad(): #torch.enable_grad():#torch.no_grad():
            mu_val_initial_conditions_raw = acq_fun.forward(X_initial_conditions_raw)

            best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                             : self.optional["NUM_RESTARTS"]
                             ].squeeze()

            X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

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

        return x_best.squeeze(dim=-2).detach(), _

