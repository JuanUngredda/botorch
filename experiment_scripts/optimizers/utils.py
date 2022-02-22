import time
from typing import Optional

import torch
from botorch.acquisition import (
    qKnowledgeGradient,
    HybridKnowledgeGradient,
    DiscreteKnowledgeGradient,
    MCKnowledgeGradient,
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


def KG_wrapper(
    method: str,
    bounds: Optional[Tensor] = None,
    num_fantasies: Optional[int] = None,
    num_discrete_points: Optional[int] = None,
    num_restarts: Optional[int] = None,
    raw_samples: Optional[int] = None,
):
    def acquisition_function(model: method):
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
            KG_acq_fun = qKnowledgeGradient(model, num_fantasies=num_fantasies)
        else:
            raise Exception(
                "method does not exist. Specify implemented method: DISCKG (Discrete KG), "
                "MCKG (Monte Carlo KG), HYBRIDKG (Hybrid KG), and ONESHOTKG (One Shot KG)"
            )
        return KG_acq_fun

    return acquisition_function
