# Available synthetic Problems:
from typing import Dict, Tuple, List, Type

import torch
from torch import dtype, DoubleTensor

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
input_dimensions = 2
Tmax = 2
lengthscale = 0.2

CONFIG_DICT = {
    "DISCKG": {
        "problems": ["GP_synthetic"],
        "method": ["DISCKG"],
        "Tmax": Tmax,
        "num_input_dim": input_dimensions,
        "lengthscale": lengthscale,
        "num_samples_initial_design": 6,
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],
        "optimization_parameters": {"num_discrete_points": ([1, 1000], torch.int)
                                    }
    },

    "MCKG": {
        "problems": ["GP_synthetic"],
        "method": ["MCKG"],
        "Tmax": Tmax,
        "num_input_dim": input_dimensions,
        "lengthscale": lengthscale,
        "num_samples_initial_design": 6,
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],
        "optimization_parameters": {"num_fantasies": ([1, 100], torch.int)
                                    }
    },

    "HYBRIDKG": {
        "problems": ["GP_synthetic"],
        "method": ["HYBRIDKG"],
        "Tmax": Tmax,
        "num_input_dim": input_dimensions,
        "lengthscale": lengthscale,
        "num_samples_initial_design": 6,
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],
        "optimization_parameters": {"num_fantasies": ([1, 100], torch.int)
                                    }
    },

    "ONESHOTKG": {
        "problems": ["GP_synthetic"],
        "method": ["ONESHOTKG"],
        "Tmax": Tmax,
        "num_input_dim": input_dimensions,
        "lengthscale": lengthscale,
        "num_samples_initial_design": 6,
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],
        "optimization_parameters": {"num_fantasies": ([1, 100], torch.int)
                                    }
    },

    "ONESHOTHYBRIDKG": {
        "problems": ["GP_synthetic"],
        "method": ["ONESHOTHYBRIDKG"],
        "Tmax": Tmax,
        "num_input_dim": input_dimensions,
        "lengthscale": lengthscale,
        "num_samples_initial_design": 6,
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],
        "optimization_parameters": {"num_fantasies": ([1, 100], torch.int)
                                    }
    },

    "RANDOM": {
        "problems": ["GP_synthetic"],
        "method": ["ONESHOTHYBRIDKG"],
        "Tmax": Tmax,
        "num_input_dim": input_dimensions,
        "lengthscale": lengthscale,
        "num_samples_initial_design": 6,
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],
        "optimization_parameters": {"num_fantasies": ([1, 100], torch.int)
                                    }
    }
}
