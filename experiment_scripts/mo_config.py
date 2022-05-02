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
    "BNH_1_scalar_experiments": {
        "problems": ["BNH"],
        "method": ["pen-maKG", "macKG"],
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 1,
        "num_samples_initial_design": 6,
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
    "SRN_1_scalar_experiments": {
        "problems": ["SRN"],
        "method": ["pen-maKG", "macKG"],
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 1,
        "num_samples_initial_design": 6,
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
    "CONSTR_1_scalar_experiments": {
        "problems": ["CONSTR"],
        "method": ["pen-maKG", "macKG"],
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 1,
        "num_samples_initial_design": 6,
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
    "ConstrainedBraninCurrin_1_scalar_experiments": {
        "problems": ["ConstrainedBraninCurrin"],
        "method": ["pen-maKG", "macKG"],
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 1,
        "num_samples_initial_design": 6,
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
    "C2DTLZ2_1_scalar_experiments": {
        "problems": ["C2DTLZ2"],
        "method": ["pen-maKG", "macKG"],
        "output_dim": 2,
        "input_dim": 3,
        "number_of_scalarizations": 1,
        "num_samples_initial_design": 8,
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
    "OSY_1_scalar_experiments": {
        "problems": ["OSY"],
        "method": ["pen-maKG", "macKG"],
        "output_dim": 2,
        "input_dim": 6,
        "number_of_scalarizations": 1,
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
    "WeldedBeam_experiments": {
        "problems": ["WeldedBeam"],
        "method": ["macKG"],
        "output_dim": 2,
        "input_dim": 4,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 10,
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
}
