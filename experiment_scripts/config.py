CONFIG_DICT = {
    "TEST1_experiments": {
        "problems": ["TEST1"],
        "method": ["DISCKG", "MCKG", "HYBRIDKG", "ONESHOTKG"],
        "num_discrete_points": [100, None, None, None],
        "num_fantasies": [None, 3, 3, 3],
        "num_restarts_inner_optimizer": [5] * 4,
        "raw_samples_inner_optimizer": [80] * 4,
        "acquisition_optimizer": ["L-BFGS-B", "Adam", "L-BFGS-B", "L-BFGS-B"],
        "num_restarts_acq_optimizer": [5] * 4,
        "raw_samples_acq_optimizer": [80] * 4,
    }
}
