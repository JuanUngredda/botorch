import time

from botorch.optim import optimize_acqf
import torch


class OptimizeAcqfAndGetDesign():

    def __init__(self, maxtime_sec, acq_func, bounds, **kwargs):
        self.nit = 0
        self.maxtime_sec = maxtime_sec
        self.acq_func = acq_func
        self.bounds = bounds

        self.raw_samples = 20 * bounds.shape[1]
        self.num_restarts = 1

    def fun(self, x_val):
        y_val = self.acq_func(x_val)
        self.x.append(x_val)
        self.y.append(y_val)
        return y_val

    def optimize(self):
        self.x = []
        self.y = []
        self.start_time = time.time()
        elapsed_time = 0
        x_final_candidate = None
        value_final_candidate = -torch.inf
        while elapsed_time < self.maxtime_sec:
            candidates = optimize_acqf(
                acq_function=self.fun,
                bounds=self.bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                return_best_only=True,
                timeout_sec=self.maxtime_sec,
            )
            self.nit += 1
            elapsed_time = time.time() - self.start_time
            # print("it-", self.nit, " Elapsed: %.3f sec" % elapsed_time)
            # print("candidate...", candidates)
            if value_final_candidate < candidates[1]:
                x_final_candidate = candidates[0].type(torch.DoubleTensor)
                value_final_candidate = candidates[1]
        return x_final_candidate, value_final_candidate
