import math
from typing import Optional

import torch
from torch import Tensor


class test_1:
    def __init__(self, base_seed: Optional[int] = None):

        self.problem = "TEST1"
        self.lb = torch.zeros((1, 2))
        self.ub = torch.ones((1, 2))

    def __call__(self, X: Tensor) -> Tensor:
        """
        test function 1
        """
        X = torch.atleast_2d(X)
        f = torch.sin(2 * math.pi * X[:, 0]) * torch.cos(2 * math.pi * X[:, 1])
        return f
