import torch
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

CONFIG_DICT = {"MCKG_GP_synthetic_dim2_l0.1": {
    "problems": ["GP_synthetic"],
    "num_input_dim": [2],
    "lengthscale": [0.1],
    "method": ["MCKG"],
    "num_samples_initial_design": [6],
    "num_max_evaluatations": [100],
    "num_discrete_points": [0],
    "num_fantasies": [2, 5, 10, 20, 50,  100, 200],
    "num_restarts_inner_optimizer": [1],
    "raw_samples_inner_optimizer": [100],
    "acquisition_optimizer": [
        "L-BFGS-B"
    ],  # "L-BFGS-B" or "Adam"
    "num_restarts_acq_optimizer": [1],
    "raw_samples_acq_optimizer": [100]},

    "ONESHOTKG_GP_synthetic_dim2_l0.1": {
    "problems": ["GP_synthetic"],
    "num_input_dim": [2],
    "lengthscale": [0.1],
    "method": ["ONESHOTKG"],
    "num_samples_initial_design": [6],
    "num_max_evaluatations": [100],
    "num_discrete_points": [0],
    "num_fantasies": [2, 5, 10, 20, 50,  100, 200],
    "num_restarts_inner_optimizer": [1],
    "raw_samples_inner_optimizer": [100],
    "acquisition_optimizer": [
        "L-BFGS-B"
    ],  # "L-BFGS-B" or "Adam"
    "num_restarts_acq_optimizer": [1],
    "raw_samples_acq_optimizer": [100]}

}

# CONFIG_DICT = {"t_ONESHOTKG_GP_synthetic_dim2_l0.1": {
#     "problems": ["GP_synthetic"],
#     "num_input_dim": [2],
#     "lengthscale": [0.1],
#     "method": ["ONESHOTKG"],
#     "num_samples_initial_design": [99],
#     "num_max_evaluatations": [100],
#     "num_discrete_points": [0],
#     "num_fantasies": [3, 10],
#     "num_restarts_inner_optimizer": [1],
#     "raw_samples_inner_optimizer": [100],
#     "acquisition_optimizer": [
#         "L-BFGS-B"
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [1],
#     "raw_samples_acq_optimizer": [100]},
#
#     "t_HYBRIDKG_GP_synthetic_dim2_l0.1": {
#     "problems": ["GP_synthetic"],
#     "num_input_dim": [2],
#     "lengthscale": [0.1],
#     "method": ["HYBRIDKG"],
#     "num_samples_initial_design": [99],
#     "num_max_evaluatations": [100],
#     "num_discrete_points": [0],
#     "num_fantasies": [3, 10],
#     "num_restarts_inner_optimizer": [1],
#     "raw_samples_inner_optimizer": [100],
#     "acquisition_optimizer": [
#         "L-BFGS-B"
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [1],
#     "raw_samples_acq_optimizer": [100]},
#
#     "t_ONESHOTHYBRIDKG_GP_synthetic_dim2_l0.1": {
#     "problems": ["GP_synthetic"],
#     "num_input_dim": [2],
#     "lengthscale": [0.1],
#     "method": ["ONESHOTHYBRIDKG"],
#     "num_samples_initial_design": [99],
#     "num_max_evaluatations": [100],
#     "num_discrete_points": [0],
#     "num_fantasies": [3,10],
#     "num_restarts_inner_optimizer": [1],
#     "raw_samples_inner_optimizer": [100],
#     "acquisition_optimizer": [
#         "L-BFGS-B"
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [1],
#     "raw_samples_acq_optimizer": [100]},
#
#     "t_MCKG_GP_synthetic_dim2_l0.1": {
#     "problems": ["GP_synthetic"],
#     "num_input_dim": [2],
#     "lengthscale": [0.1],
#     "method": ["MCKG"],
#     "num_samples_initial_design": [99],
#     "num_max_evaluatations": [100],
#     "num_discrete_points": [0],
#     "num_fantasies": [3, 10],
#     "num_restarts_inner_optimizer": [1],
#     "raw_samples_inner_optimizer": [100],
#     "acquisition_optimizer": [
#         "L-BFGS-B"
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [1],
#     "raw_samples_acq_optimizer": [100]},
#
#     "t_DISCKG_GP_synthetic_dim2_l0.1": {
#     "problems": ["GP_synthetic"],
#     "num_input_dim": [2],
#     "lengthscale": [0.1],
#     "method": ["DISCKG"],
#     "num_samples_initial_design": [99],
#     "num_max_evaluatations": [100],
#     "num_discrete_points": [3, 1000],
#     "num_fantasies": [0],
#     "num_restarts_inner_optimizer": [1],
#     "raw_samples_inner_optimizer": [100],
#     "acquisition_optimizer": [
#         "L-BFGS-B"
#     ],  # "L-BFGS-B" or "Adam"
#     "num_restarts_acq_optimizer": [1],
#     "raw_samples_acq_optimizer": [100]},
# }