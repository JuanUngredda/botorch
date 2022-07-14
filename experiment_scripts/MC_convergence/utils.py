import time
from typing import Optional

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

def acq_values_recorder(method):
    def output_recorderd(*args, **kw):
        result = method(*args, **kw)
        if "log_acq_vals" in kw:
            kw["log_acq_vals"].append(result[-1])
        return result
    return output_recorderd


class RandomSample():
    def __init__(self, dim=int):
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:

        return torch.rand((1, self.dim))

def KG_wrapper(
    method: str,
    bounds: Optional[Tensor] = None,
    num_fantasies: Optional[int] = None,
    num_discrete_points: Optional[int] = None,
    num_restarts: Optional[int] = None,
    raw_samples: Optional[int] = None
):
    def acquisition_function(model: method, x_optimiser: Optional[Tensor]=None, current_value: Optional[Tensor]=None):

        if method == "DISCKG":

            X_discretisation = torch.rand(size=(num_discrete_points, 1, bounds.shape[1]))

            KG_acq_fun = DiscreteKnowledgeGradient(
                model=model,
                bounds=bounds,
                num_discrete_points=num_discrete_points,
                X_discretisation=X_discretisation,
                current_optimiser = x_optimiser
            )
        elif method == "MCKG":

            KG_acq_fun = MCKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts if num_restarts is not None else 4,
                raw_samples=raw_samples if raw_samples is not None else 80
                , current_value=current_value
            )
        elif method == "HYBRIDKG":
            KG_acq_fun = HybridKnowledgeGradient(
                model,
                bounds=bounds,
                num_fantasies=num_fantasies,
                num_restarts=num_restarts if num_restarts is not None else 4,
                raw_samples=raw_samples if raw_samples is not None else 80,
            )
        elif method == "ONESHOTKG":
            KG_acq_fun = qKnowledgeGradient(model, num_fantasies=num_fantasies, current_value= current_value)

        elif method =="RANDOMKG":
            KG_acq_fun = RandomSample(dim=bounds.shape[1])

        elif method=="ONESHOTHYBRIDKG":
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
