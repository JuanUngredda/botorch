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

# 2, 5, 10, 20, 50,  100, 200
CONFIG_DICT = {

    "MCKG_Branin_2": {
        "problems": ["Branin"],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "MCKG_Branin_10": {
        "problems": ["Branin"],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "HYBRIDKG_Branin_2": {
        "problems": ["Branin"],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "HYBRIDKG_Branin_10": {
        "problems": ["Branin"],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "DiscreteKG_Branin_2": {
        "problems": ["Branin"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "DiscreteKG_Branin_1000": {
        "problems": ["Branin"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "DiscreteKG_Rosenbrock_2": {
        "problems": ["Rosenbrock"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "DiscreteKG_Rosenbrock_1000": {
        "problems": ["Rosenbrock"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "MCKG_Rosenbrock_2": {
        "problems": ["Rosenbrock"],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "MCKG_Rosenbrock_10": {
        "problems": ["Rosenbrock"],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "HYBRIDKG_Rosenbrock_2": {
        "problems": ["Rosenbrock"],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "HYBRIDKG_Rosenbrock_10": {
        "problems": ["Rosenbrock"],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "DiscreteKG_Hartmann_2": {
        "problems": ["Hartmann"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "DiscreteKG_Hartmann_1000": {
        "problems": ["Hartmann"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "MCKG_Hartmann_2": {
        "problems": ["Hartmann"],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "MCKG_Hartmann_10": {
        "problems": ["Hartmann"],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "HYBRIDKG_Hartmann_2": {
        "problems": ["Hartmann"],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "HYBRIDKG_Hartmann_10": {
        "problems": ["Hartmann"],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    },
    "DiscreteKG_Hartmann_3D_2": {
        "problems": ["Hartmann"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [10],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80]},

    "DiscreteKG_Hartmann_3D_1000": {
        "problems": ["Hartmann"],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [50],
        "num_discrete_points": [1000],
        "num_fantasies": [None],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [3],
        "raw_samples_acq_optimizer": [80],
    }
}
# },
# "Branin_experiments_5": {
#     "problems": ["Branin"],
#     "method": ["DISCKG", "MCKG", "HYBRIDKG", "ONESHOTKG"],
#     "num_samples_initial_design": [10]*4,
#     "num_max_evaluatations":[50]*4,
#     "num_discrete_points": [5, None, None, None],
#     "num_fantasies": [None, 5, 5, 5],
#     "num_restarts_inner_optimizer": [1] * 4,
#     "raw_samples_inner_optimizer": [100] * 4,
#     "acquisition_optimizer": [
#         "L-BFGS-B",
#         "L-BFGS-B",
#         "L-BFGS-B",
#         "L-BFGS-B",
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [3] * 4,
#     "raw_samples_acq_optimizer": [80] * 4,
# },
# "Rosenbrock_experiments_5": {
#     "problems": ["Rosenbrock"],
#     "method": ["DISCKG", "MCKG", "HYBRIDKG", "ONESHOTKG"],
#     "num_samples_initial_design": [10]*4,
#     "num_max_evaluatations": [50]*4,
#     "num_discrete_points": [5, None, None, None],
#     "num_fantasies": [None, 5, 5, 5],
#     "num_restarts_inner_optimizer": [1] * 4,
#     "raw_samples_inner_optimizer": [100] * 4,
#     "acquisition_optimizer": [
#         "L-BFGS-B",
#         "L-BFGS-B",
#         "L-BFGS-B",
#         "L-BFGS-B",
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [3] * 4,
#     "raw_samples_acq_optimizer": [80] * 4,
# },
