# Available synthetic Problems:

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

CONFIG_DICT = {
    "Branin_experiments": {
        "problems": ["Branin"],
        "method": ["DISCKG", "MCKG", "HYBRIDKG", "ONESHOTKG"],
        "num_discrete_points": [1000, None, None, None],
        "num_fantasies": [5, 5, 5, 5],
        "num_restarts_inner_optimizer": [1] * 4,
        "raw_samples_inner_optimizer": [100] * 4,
        "acquisition_optimizer": [
            "L-BFGS-B",
            "L-BFGS-B",
            "L-BFGS-B",
            "L-BFGS-B",
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3] * 4,
        "raw_samples_acq_optimizer": [80] * 4,
    },
    "Rosenbrock_experiments": {
        "problems": ["Rosenbrock"],
        "method": ["DISCKG", "MCKG", "HYBRIDKG", "ONESHOTKG"],
        "num_discrete_points": [1000, None, None, None],
        "num_fantasies": [5, 5, 5, 5],
        "num_restarts_inner_optimizer": [1] * 4,
        "raw_samples_inner_optimizer": [100] * 4,
        "acquisition_optimizer": [
            "L-BFGS-B",
            "L-BFGS-B",
            "L-BFGS-B",
            "L-BFGS-B",
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3] * 4,
        "raw_samples_acq_optimizer": [80] * 4,
    },
}
