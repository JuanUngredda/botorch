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
    "rosenbrock_experiments": {
        "problems": ["Rosenbrock"],
        "method": ["DISCKG", "MCKG", "HYBRIDKG", "ONESHOTKG"],
        "num_discrete_points": [1000, None, None, None],
        "num_fantasies": [None, 5, 5, 5],
        "num_restarts_inner_optimizer": [5] * 4,
        "raw_samples_inner_optimizer": [80] * 4,
        "acquisition_optimizer": ["L-BFGS-B", "Adam", "L-BFGS-B", "L-BFGS-B"],
        "num_restarts_acq_optimizer": [5] * 4,
        "raw_samples_acq_optimizer": [80] * 4,
    }
}
