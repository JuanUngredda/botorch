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
    "BNH_mc_experiments": {
        "problems": ["BNH"],
        "method": ["macKG"],
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": [3,10,50,100],
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 7,
        "utility_model": [ "Tche"],
        "num_discrete_points": [150],
        "num_fantasies": [3,10,50,100],
        "num_restarts_inner_optimizer": 1 ,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    }
}
