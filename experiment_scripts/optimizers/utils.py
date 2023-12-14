import time
from typing import Optional
from torch.autograd import Variable
import torch
from botorch.acquisition import (
    qKnowledgeGradient,
    HybridKnowledgeGradient,
    DiscreteKnowledgeGradient,
    MCKnowledgeGradient,
    HybridOneShotKnowledgeGradient
)
from torch import Tensor


#################################################################
#                                                               #
#                           LHC SAMPLERS                        #
#                                                               #
#################################################################


def lhc(
        n: int,
        dim: Optional[int] = None,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
) -> Tensor:
    """
    Parameters
    ----------
    n: sample size
    dim: optional, dimenions of the cube
    lb: lower bound, Tensor
    ub: upper bound, Tensor
    Returns
    -------
    x: Tensor, shape (n, dim)
    """

    if dim is not None:
        assert (lb is None) and (ub is None), "give dim OR bounds"
        lb = torch.zeros(dim, dtype=torch.double)
        ub = torch.ones(dim, dtype=torch.double)

    else:
        assert (lb is not None) and (ub is not None), "give dim OR bounds"
        lb = lb.squeeze()
        ub = ub.squeeze()
        dim = len(lb)
        assert len(lb) == len(ub), f"bounds are not same shape:{len(lb)}!={len(ub)}"

    x = torch.zeros((n, dim), dtype=torch.double)
    if n > 0:
        for d in range(dim):
            x[:, d] = (torch.randperm(n) + torch.rand(n)) * (1 / n)
            x[:, d] = (ub[d] - lb[d]) * x[:, d] + lb[d]

    return x


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            if name in kw["log_time"].keys():
                kw["log_time"][name].append((te - ts))
            else:
                kw["log_time"][name] = [te - ts]

        return result

    return timed


class RandomSample():
    def __init__(self, dim=int):
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        return torch.rand((1, self.dim))


def KG_Objective_Function(model,
                          method: str,
                          bounds: Optional[Tensor] = None,
                          x_optimiser: Optional[Tensor] = None,
                          current_value: Optional[Tensor] = None):
    def acquisition_function(**kwargs):

        if "num_discrete_points" in kwargs:
            num_discrete_points = int(kwargs["num_discrete_points"])

        if "num_fantasies" in kwargs:
            num_fantasies = int(kwargs["num_fantasies"])

        if "raw_samples_internal_optimizer" in kwargs and "proportion_restarts_internal_optimizer" in kwargs:
            raw_samples_internal_optimizer = int(kwargs["raw_samples_internal_optimizer"])
            proportion = float(kwargs["proportion_restarts_internal_optimizer"])
            num_restarts_internal_optimizer = max(1, int(raw_samples_internal_optimizer * proportion))

        if method == "DISCKG":
            print(kwargs)
            KG_acq_fun = DiscreteKnowledgeGradient(
                model=model,
                bounds=bounds,
                num_discrete_points=num_discrete_points,
                X_discretisation=None,
            )
        elif method == "MCKG":

            KG_acq_fun = MCKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts_internal_optimizer if num_restarts_internal_optimizer is not None else 4,
                raw_samples=raw_samples_internal_optimizer if raw_samples_internal_optimizer is not None else 80,
            )
        elif method == "HYBRIDKG":
            KG_acq_fun = HybridKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts_internal_optimizer if num_restarts_internal_optimizer is not None else 4,
                raw_samples=raw_samples_internal_optimizer if raw_samples_internal_optimizer is not None else 80,
            )
        elif method == "ONESHOTKG":

            KG_acq_fun = qKnowledgeGradient(model,
                                            num_fantasies=num_fantasies,
                                            current_value=current_value)
        elif method == "RANDOMKG":
            KG_acq_fun = RandomSample(dim=bounds.shape[1])

        elif method == "ONESHOTHYBRIDKG":
            KG_acq_fun = HybridOneShotKnowledgeGradient(model=model,
                                                        num_fantasies=num_fantasies,
                                                        x_optimiser=x_optimiser)
        else:
            raise Exception(
                "method does not exist. Specify implemented method: DISCKG (Discrete KG), "
                "MCKG (Monte Carlo KG), HYBRIDKG (Hybrid KG), and ONESHOTKG (One Shot KG)"
            )
        return KG_acq_fun

    return acquisition_function


class TmaxWrapper():
    """
    This wrapper measures the evaluation time of a function and stops it when a time limit, tmax, is reached.
    """

    def __init__(self, fun, tmax):
        self.final_difference_evaluation = None
        self.time_difference = None
        self.fun = fun
        self.start_time = time.time()
        self.current_time = time.time()

        assert isinstance(tmax, float)
        self.tmax = tmax
        self.x = []
        self.y = []

    def fun_evaluate(self, x):
        self.current_time = time.time()
        y_vals = []
        for idx, x_val in enumerate(x):
            y_val = self.fun(x_val)
            y_vals.append(y_val)
            self.current_time = time.time()
            difference_post_evaluation = self.current_time - self.start_time
            if difference_post_evaluation < self.tmax:
                self.time_difference = difference_post_evaluation
                self.x.append(x_val)
                self.y.append(y_val)
        self.final_difference_evaluation = self.current_time - self.start_time
        return torch.concat(y_vals)

    def get_time_difference(self):
        return self.time_difference

    def get_time_ratio(self):
        return self.final_difference_evaluation / self.time_difference

    def get_best_x(self):
        y_tensor = torch.Tensor(self.y)
        return self.x[torch.argmax(y_tensor)]

    def get_best_y(self):
        y_tensor = torch.Tensor(self.y)
        return torch.max(y_tensor)


def KG_wrapper(
        method: str,
        bounds: Optional[Tensor] = None,
        num_fantasies: Optional[int] = None,
        num_discrete_points: Optional[int] = None,
        num_restarts: Optional[int] = None,
        raw_samples: Optional[int] = None
):
    def acquisition_function(model: method,
                             x_best: Optional[Tensor] = None,
                             fn_best: Optional[Tensor] = None):
        if method == "DISCKG":

            KG_acq_fun = DiscreteKnowledgeGradient(
                model=model,
                bounds=bounds,
                num_discrete_points=num_discrete_points,
                X_discretisation=None,
            )
        elif method == "MCKG":

            KG_acq_fun = MCKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts if num_restarts is not None else 4,
                raw_samples=raw_samples if raw_samples is not None else 80,
                current_optimiser=x_best
            )
        elif method == "HYBRIDKG":
            KG_acq_fun = HybridKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts if num_restarts is not None else 4,
                raw_samples=raw_samples if raw_samples is not None else 80,
                current_optimiser=x_best
            )
        elif method == "ONESHOTKG":

            KG_acq_fun = qKnowledgeGradient(model,
                                            num_fantasies=num_fantasies,
                                            current_value=fn_best)

        elif method == "RANDOMKG":
            KG_acq_fun = RandomSample(dim=bounds.shape[1])

        elif method == "ONESHOTHYBRIDKG":
            KG_acq_fun = HybridOneShotKnowledgeGradient(model=model,
                                                        num_fantasies=num_fantasies,
                                                        x_optimiser=x_best)
        else:
            raise Exception(
                "method does not exist. Specify implemented method: DISCKG (Discrete KG), "
                "MCKG (Monte Carlo KG), HYBRIDKG (Hybrid KG), and ONESHOTKG (One Shot KG)"
            )
        return KG_acq_fun

    return acquisition_function
