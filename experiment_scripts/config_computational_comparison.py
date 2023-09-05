# Available synthetic Problems:
import torch
# 2D problems:

# "Egg-holder"
# "Sum of Powers"
# "Branin"
# "Cosines"
# "Mccormick"
# "Goldstein"
# "Six-hump camel"
# "dropwave"
# "Rosenbrock"
# "beale"

# 2, 5, 10, 20, 50,  100, 200

true_underlying_config_params = {
    "num_fantasies": 5,
    "proportion_restarts_internal_optimizer": 0.2,
    "raw_samples_internal_optimizer": 5}

CONFIG_DICT = {
    "DISCKG_dim2": {
        "problems": ["GP_synthetic"],
        "method": ["DISCKG"],
        "Tmax": [10.0],
        "num_input_dim": [2],
        "lengthscale": [0.1],
        "num_samples_initial_design": [6],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],
        "optimization_parameters": {"num_discrete_points": ([10, 12], torch.int),
                                    "proportion_restarts_internal_optimizer": ([0, 1], torch.DoubleTensor),
                                    "raw_samples_internal_optimizer": ([1, 8], torch.int),
                                    "proportion_restarts_external_optimizer": ([0, 1], torch.DoubleTensor),
                                    "raw_samples_external_optimizer": ([15, 20], torch.int)
                                    }
    }}
