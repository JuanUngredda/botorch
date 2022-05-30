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

CONFIG_DICT = {"ONESHOTHYBRIDKG_GP_synthetic_10_dim2_l0.1": {
    "problems": ["GP_synthetic"],
    "num_input_dim": [2],
    "lengthscale": [0.1],
    "method": ["ONESHOTHYBRIDKG"],
    "num_samples_initial_design": [6],
    "num_max_evaluatations": [100],
    "num_discrete_points": [1000],
    "num_fantasies": [10],
    "num_restarts_inner_optimizer": [1],
    "raw_samples_inner_optimizer": [100],
    "acquisition_optimizer": [
        "L-BFGS-B"
    ],  # "L-BFGS-B" or "Adam"
    "num_restarts_acq_optimizer": [1],
    "raw_samples_acq_optimizer": [50]},
    "ONESHOTHYBRIDKG_GP_synthetic_10_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["ONESHOTHYBRIDKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "ONESHOTHYBRIDKG_GP_synthetic_10_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["ONESHOTHYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "ONESHOTHYBRIDKG_GP_synthetic_10_dim4_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.4],
        "method": ["ONESHOTHYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "ONESHOTHYBRIDKG_GP_synthetic_3_dim2_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.1],
        "method": ["ONESHOTHYBRIDKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "ONESHOTHYBRIDKG_GP_synthetic_3_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["ONESHOTHYBRIDKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "ONESHOTHYBRIDKG_GP_synthetic_3_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["ONESHOTHYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "ONESHOTHYBRIDKG_GP_synthetic_3_dim4_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.4],
        "method": ["ONESHOTHYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "HYBRIDKG_GP_synthetic_10_dim2_l0.1": {
    "problems": ["GP_synthetic"],
    "num_input_dim": [2],
    "lengthscale": [0.1],
    "method": ["HYBRIDKG"],
    "num_samples_initial_design": [6],
    "num_max_evaluatations": [100],
    "num_discrete_points": [1000],
    "num_fantasies": [10],
    "num_restarts_inner_optimizer": [1],
    "raw_samples_inner_optimizer": [100],
    "acquisition_optimizer": [
        "L-BFGS-B"
    ],  # "L-BFGS-B" or "Adam"
    "num_restarts_acq_optimizer": [1],
    "raw_samples_acq_optimizer": [50]},
    "HYBRIDKG_GP_synthetic_10_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "HYBRIDKG_GP_synthetic_10_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "HYBRIDKG_GP_synthetic_10_dim4_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.4],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]

    },

    "HYBRIDKG_GP_synthetic_3_dim2_l0.1": {
    "problems": ["GP_synthetic"],
    "num_input_dim": [2],
    "lengthscale": [0.1],
    "method": ["HYBRIDKG"],
    "num_samples_initial_design": [6],
    "num_max_evaluatations": [100],
    "num_discrete_points": [1000],
    "num_fantasies": [3],
    "num_restarts_inner_optimizer": [1],
    "raw_samples_inner_optimizer": [100],
    "acquisition_optimizer": [
        "L-BFGS-B"
    ],  # "L-BFGS-B" or "Adam"
    "num_restarts_acq_optimizer": [1],
    "raw_samples_acq_optimizer": [50]},
    "HYBRIDKG_GP_synthetic_3_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "HYBRIDKG_GP_synthetic_3_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "HYBRIDKG_GP_synthetic_3_dim4_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.4],
        "method": ["HYBRIDKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "MCKG_GP_synthetic_10_dim2_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.1],
        "method": ["MCKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "MCKG_GP_synthetic_10_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["MCKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "MCKG_GP_synthetic_10_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [10],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "MCKG_GP_synthetic_10_dim4_l0.4": {
            "problems": ["GP_synthetic"],
            "num_input_dim": [4],
            "lengthscale": [0.4],
            "method": ["MCKG"],
            "num_samples_initial_design": [10],
            "num_max_evaluatations": [100],
            "num_discrete_points": [1000],
            "num_fantasies": [10],
            "num_restarts_inner_optimizer": [1],
            "raw_samples_inner_optimizer": [100],
            "acquisition_optimizer": [
                "L-BFGS-B"
            ],  # "L-BFGS-B" or "Adam"
            "num_restarts_acq_optimizer": [1],
            "raw_samples_acq_optimizer": [50]},

    "MCKG_GP_synthetic_3_dim2_l0.1": {
    "problems": ["GP_synthetic"],
    "num_input_dim": [2],
    "lengthscale": [0.1],
    "method": ["MCKG"],
    "num_samples_initial_design": [6],
    "num_max_evaluatations": [100],
    "num_discrete_points": [1000],
    "num_fantasies": [3],
    "num_restarts_inner_optimizer": [1],
    "raw_samples_inner_optimizer": [100],
    "acquisition_optimizer": [
        "L-BFGS-B"
    ],  # "L-BFGS-B" or "Adam"
    "num_restarts_acq_optimizer": [1],
    "raw_samples_acq_optimizer": [50]},
    "MCKG_GP_synthetic_3_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["MCKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "MCKG_GP_synthetic_3_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "MCKG_GP_synthetic_3_dim4_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.4],
        "method": ["MCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [3],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "DISCKG_GP_synthetic_1000_dim2_l0.1": {
    "problems": ["GP_synthetic"],
    "num_input_dim": [2],
    "lengthscale": [0.1],
    "method": ["DISCKG"],
    "num_samples_initial_design": [6],
    "num_max_evaluatations": [100],
    "num_discrete_points": [1000],
    "num_fantasies": [2],
    "num_restarts_inner_optimizer": [1],
    "raw_samples_inner_optimizer": [100],
    "acquisition_optimizer": [
        "L-BFGS-B"
    ],  # "L-BFGS-B" or "Adam"
    "num_restarts_acq_optimizer": [1],
    "raw_samples_acq_optimizer": [50]},
    "DISCKG_GP_synthetic_1000_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["DISCKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "DISCKG_GP_synthetic_1000_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "DISCKG_GP_synthetic_1000_dim4_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.4],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [1000],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "DISCKG_GP_synthetic_3_dim2_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.1],
        "method": ["DISCKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [3],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "DISCKG_GP_synthetic_3_dim2_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [2],
        "lengthscale": [0.4],
        "method": ["DISCKG"],
        "num_samples_initial_design": [6],
        "num_max_evaluatations": [100],
        "num_discrete_points": [3],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},

    "DISCKG_GP_synthetic_3_dim4_l0.1": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.1],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [3],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
    "DISCKG_GP_synthetic_3_dim4_l0.4": {
        "problems": ["GP_synthetic"],
        "num_input_dim": [4],
        "lengthscale": [0.4],
        "method": ["DISCKG"],
        "num_samples_initial_design": [10],
        "num_max_evaluatations": [100],
        "num_discrete_points": [3],
        "num_fantasies": [2],
        "num_restarts_inner_optimizer": [1],
        "raw_samples_inner_optimizer": [100],
        "acquisition_optimizer": [
            "L-BFGS-B"
        ],  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": [1],
        "raw_samples_acq_optimizer": [50]},
}


# CONFIG_DICT = {"RANDOMKG_GP_synthetic_dim2_l0.1": {
#     "problems": ["GP_synthetic"],
#     "num_input_dim": [2],
#     "lengthscale": [0.1],
#     "method": ["RANDOMKG"],
#     "num_samples_initial_design": [6],
#     "num_max_evaluatations": [100],
#     "num_discrete_points": [10],
#     "num_fantasies": [2],
#     "num_restarts_inner_optimizer": [1],
#     "raw_samples_inner_optimizer": [100],
#     "acquisition_optimizer": [
#         "L-BFGS-B"
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [3],
#     "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim2_l0.4": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [2],
#         "lengthscale": [0.4],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [6],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim2_l1": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [2],
#         "lengthscale": [1],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [6],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim4_l0.1": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [4],
#         "lengthscale": [0.1],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [10],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim4_l0.4": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [4],
#         "lengthscale": [0.4],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [10],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim4_l1": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [4],
#         "lengthscale": [1],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [10],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim6_l0.1": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [6],
#         "lengthscale": [0.1],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [10],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim6_l0.4": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [6],
#         "lengthscale": [0.4],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [10],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
#     "RANDOMKG_GP_synthetic_dim6_l1": {
#         "problems": ["GP_synthetic"],
#         "num_input_dim": [6],
#         "lengthscale": [1],
#         "method": ["RANDOMKG"],
#         "num_samples_initial_design": [10],
#         "num_max_evaluatations": [100],
#         "num_discrete_points": [10],
#         "num_fantasies": [2],
#         "num_restarts_inner_optimizer": [1],
#         "raw_samples_inner_optimizer": [100],
#         "acquisition_optimizer": [
#             "L-BFGS-B"
#         ],  # "L-BFGS-B" or "Adam"
#         "num_restarts_acq_optimizer": [3],
#         "raw_samples_acq_optimizer": [80]},
# }

# CONFIG_DICT = {
#             "RANDOMKG_Branin": {
#                 "problems": ["Branin"],
#                 "num_input_dim": [2] ,
#                 "lengthscale": [2] ,
#                 "method": ["RANDOMKG"],
#                 "num_samples_initial_design": [10],
#                 "num_max_evaluatations": [50],
#                 "num_discrete_points": [10],
#                 "num_fantasies": [2],
#                 "num_restarts_inner_optimizer": [1],
#                 "raw_samples_inner_optimizer": [100],
#                 "acquisition_optimizer": [
#                     "L-BFGS-B"
#                 ],  # "L-BFGS-B" or "Adam"
#                 "num_restarts_acq_optimizer": [3],
#                 "raw_samples_acq_optimizer": [80]},
#
#                   "MCKG_Branin_2": {
#                       "problems": ["Branin"],
#                       "method": ["MCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [2],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "MCKG_Branin_10": {
#                       "problems": ["Branin"],
#                       "method": ["MCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [10],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "HYBRIDKG_Branin_2": {
#                       "problems": ["Branin"],
#                       "method": ["HYBRIDKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [2],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "HYBRIDKG_Branin_10": {
#                       "problems": ["Branin"],
#                       "method": ["HYBRIDKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [10],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "DISCKG_Branin_2": {
#                       "problems": ["Branin"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "DISCKG_Branin_1000": {
#                       "problems": ["Branin"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "DISCKG_Rosenbrock_2": {
#                       "problems": ["Rosenbrock"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "DISCKG_Rosenbrock_1000": {
#                       "problems": ["Rosenbrock"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "MCKG_Rosenbrock_2": {
#                       "problems": ["Rosenbrock"],
#                       "method": ["MCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [2],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "MCKG_Rosenbrock_10": {
#                       "problems": ["Rosenbrock"],
#                       "method": ["MCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [10],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "HYBRIDKG_Rosenbrock_2": {
#                       "problems": ["Rosenbrock"],
#                       "method": ["HYBRIDKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [2],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "HYBRIDKG_Rosenbrock_10": {
#                       "problems": ["Rosenbrock"],
#                       "method": ["HYBRIDKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [10],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "DISCKG_Hartmann_2": {
#                       "problems": ["Hartmann"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "DISCKG_Hartmann_1000": {
#                       "problems": ["Hartmann"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "MCKG_Hartmann_2": {
#                       "problems": ["Hartmann"],
#                       "method": ["MCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [2],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "MCKG_Hartmann_10": {
#                       "problems": ["Hartmann"],
#                       "method": ["MCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [10],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "HYBRIDKG_Hartmann_2": {
#                       "problems": ["Hartmann"],
#                       "method": ["HYBRIDKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [2],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "HYBRIDKG_Hartmann_10": {
#                       "problems": ["Hartmann"],
#                       "method": ["HYBRIDKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [10],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "DISCKG_Hartmann_3D_2": {
#                       "problems": ["Hartmann"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "DISCKG_Hartmann_3D_1000": {
#                       "problems": ["Hartmann"],
#                       "method": ["DISCKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [1000],
#                       "num_fantasies": [None],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80],
#                   },
#                   "ONESHOTKG_Branin_2": {
#                       "problems": ["Branin"],
#                       "method": ["ONESHOTKG"],
#                       "num_samples_initial_design": [10],
#                       "num_max_evaluatations": [50],
#                       "num_discrete_points": [10],
#                       "num_fantasies": [2],
#                       "num_restarts_inner_optimizer": [1],
#                       "raw_samples_inner_optimizer": [100],
#                       "acquisition_optimizer": [
#                           "L-BFGS-B"
#                       ],  # "L-BFGS-B" or "Adam"
#                       "num_restarts_acq_optimizer": [3],
#                       "raw_samples_acq_optimizer": [80]},
#
#                   "ONESHOTKG_Branin_10": {
#                                     "problems": ["Branin"],
#                                     "method": ["ONESHOTKG"],
#                                     "num_samples_initial_design": [10],
#                                     "num_max_evaluatations": [50],
#                                     "num_discrete_points": [10],
#                                     "num_fantasies": [10],
#                                     "num_restarts_inner_optimizer": [1],
#                                     "raw_samples_inner_optimizer": [100],
#                                     "acquisition_optimizer": [
#                                         "L-BFGS-B"
#                                     ],  # "L-BFGS-B" or "Adam"
#                                     "num_restarts_acq_optimizer": [3],
#                                     "raw_samples_acq_optimizer": [80]},
#
#                     "ONESHOTKG_Branin_125": {
#                         "problems": ["Branin"],
#                         "method": ["ONESHOTKG"],
#                         "num_samples_initial_design": [10],
#                         "num_max_evaluatations": [50],
#                         "num_discrete_points": [10],
#                         "num_fantasies": [125],
#                         "num_restarts_inner_optimizer": [1],
#                         "raw_samples_inner_optimizer": [100],
#                         "acquisition_optimizer": [
#                             "L-BFGS-B"
#                         ],  # "L-BFGS-B" or "Adam"
#                         "num_restarts_acq_optimizer": [3],
#                         "raw_samples_acq_optimizer": [80]},
#                                                 }
