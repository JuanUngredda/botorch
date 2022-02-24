import logging
import sys
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .utils import lhc

LOG_FORMAT = (
    "%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s:  %(message)s"
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


###################################################################
##                                                               ##
##                         OPTIMIZERS                            ##
##                                                               ##
###################################################################

# All optimizers must have the following components:
#   - def __init__(test_fun, budget, *kwargs)
#   - def optimize(): tries to learn optima
#   - def get_next_point(): find (s,x) to evaluate for next iteration
#   - def policy(): returns recommended action
#   - def test(): evaluates policy() action averaged over many seeds.


class BaseOptimizer(ABC):
    """
    Randomly picks a new point to sample at each time step
    and at test time, when given a test state, it takes the nearest 10%
    of sample states, finds the state with the best y value and returns x.
    """

    def __init__(
        self,
        fun,
        lb: Tensor,
        ub: Tensor,
        n_max: int,
        n_init: int = 20,
        ns0: int = None,
    ):
        """
        ARGS:
            fun: expensive black box function: X x {1,2,3,..} -> R
            lb: np.ndarray, lower bounds on x
            ub: np.ndarray, upper bounds on x
            n_max: int, number of call to tet funtion
            n_init: int, number of samples to start the BO
            ns0: int, number of seeds to sample, default n_init
        RETURNS:
            Optimizer object
        """
        logger.info(f"Initializing {type(self)}")

        if ns0 is None:
            self.ns0 = n_init
        else:
            self.ns0 = ns0
        self.n_init = n_init
        self.n_max = n_max
        self.f = fun
        self.bounds = fun.bounds
        self.lb = lb.squeeze(-1)
        self.ub = ub.squeeze(-1)
        self.dim = len(lb.squeeze())
        self.performance = torch.zeros((0, 2))
        self.method_time = {}
        self.gp_likelihood_noise = torch.Tensor([])
        self.gp_lengthscales = torch.Tensor([])
        # no need to test every step, 30 points will be enough for a results plot.
        self.testable_iters = torch.unique(
            torch.linspace(n_init, n_max, steps=31, dtype=int)
        )
        logger.info("Testable iters: %s", self.testable_iters)

    def optimize(self):

        logger.info(f"Starting optim, n_init: {self.n_init}")

        # initial random dataset
        self.x_train = lhc(self.n_init, dim=self.dim) # Tensor (n_init , X_dim)
        self.y_train = torch.Tensor(
            [self.evaluate_objective(x_i) for x_i in self.x_train]
        ).reshape((self.x_train.shape[0], 1)) # Tensor (n_init , 1)

        # test initial
        self.test()
        logger.info("Test performance:\n %s", self.performance[-1, :])

        # start iterating until the budget is exhausted.
        for _ in range(self.n_max - self.n_init):

            # collect next points
            x_new = self.get_next_point()
            y_new = self.evaluate_objective(x_new)

            # update stored data
            self.x_train = torch.vstack([self.x_train, x_new.reshape(1, -1)])
            self.y_train = torch.vstack((self.y_train, y_new))

            logger.info(f"Running optim, n: {self.x_train.shape[0]}")

            # test if necessary
            if torch.any(len(self.y_train) == self.testable_iters):
                _ = self.test()
                logger.info(f"Test performance: {self.performance[-1, :]}")

    def evaluate_objective(self, x: Tensor, **kwargs) -> Tensor:
        """
        evaluate objective function f(x)
        """

    @abstractmethod
    def get_next_point(self):
        """
        return next design from acquisition function (acq) as, x=argmax_{x} acq(x).
        """

    @abstractmethod
    def policy(self):
        """
        Return the recommended x value
        """

    def save(self):
        """
        saves intermediate results in directory.
        """

    def test(self):
        x_rec = self.policy()
        y_true = self.evaluate_objective(x=x_rec, log_time=self.method_time)
        n = len(self.y_train) * 1.0
        self.performance = torch.vstack([self.performance, torch.Tensor([n, y_true])])

        self.save()
        return y_true
