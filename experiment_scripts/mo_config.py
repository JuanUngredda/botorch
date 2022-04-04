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
    "OSY_experiments": {
        "problems": ["OSY"],
        "method": ["macKG"],
        "output_dim": 2,
        "input_dim": 6,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 14,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 20,
    },
    "C2DTLZ2_experiments": {
        "problems": ["C2DTLZ2"],
        "method": ["macKG"],
        "output_dim": 2,
        "input_dim": 3,
        "number_of_scalarizations": 10 ,
        "num_samples_initial_design": 10,
        "num_max_evaluatations": 100,
        "utility_model": [ "Tche"],
        "num_discrete_points": 5,
        "num_fantasies": 5,
        "num_restarts_inner_optimizer": 1 ,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 20,
    },
    "BraninCurrin_experiments": {
        "problems": ["ConstrainedBraninCurrin"],
        "method": ["macKG"],
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 10,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 5,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 20,
    },
}
