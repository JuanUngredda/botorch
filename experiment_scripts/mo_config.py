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
    "BNH_macKG_experiments": {
        "problems": ["BNH"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 20,
    },
    "BNH_benchmarks_experiments": {
        "problems": ["BNH"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },

    "noisy_BNH_macKG_experiments": {
        "problems": ["BNH"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": 1,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 20,
    },
    "noisy_BNH_benchmarks_experiments": {
        "problems": ["BNH"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": 1,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },

    "SRN_macKG_experiments": {
        "problems": ["SRN"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 20,
    },
    "SRN_benchmarks_experiments": {
        "problems": ["SRN"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },

    "noisy_SRN_macKG_experiments": {
        "problems": ["SRN"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },
    "noisy_SRN_benchmarks_experiments": {
        "problems": ["SRN"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },
    "ConstrainedBraninCurrin_macKG_experiments": {
        "problems": ["ConstrainedBraninCurrin"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },
    "ConstrainedBraninCurrin_benchmarks_experiments": {
        "problems": ["ConstrainedBraninCurrin"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },

    "CONSTR_macKG_experiments": {
        "problems": ["CONSTR"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },
    "CONSTR_benchmarks_experiments": {
        "problems": ["CONSTR"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 2,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },

    "C2DTLZ2_macKG_experiments": {
        "problems": ["C2DTLZ2"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 3,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 8,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },
    "C2DTLZ2_benchmarks_experiments": {
        "problems": ["C2DTLZ2"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 3,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 8,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },

    "OSY_macKG_experiments": {
        "problems": ["OSY"],
        "method": ["macKG", "pen-maKG"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 6,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 14,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 3,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },
    "OSY_benchmarks_experiments": {
        "problems": ["OSY"],
        "method": ["cEHI", "cParEGO"],
        "noise_lvl": None,
        "output_dim": 2,
        "input_dim": 6,
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 14,
        "num_max_evaluatations": 100,
        "utility_model": ["Tche"],
        "num_discrete_points": 50,
        "num_fantasies": 128,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "L-BFGS-B",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    }

}
