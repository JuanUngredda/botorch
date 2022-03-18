import argparse
import os
import subprocess as sp

import numpy as np


# This is a bare script that receives args, prints something, wastes some time,function_caller_test_func_2_TS
# and saves something. Use this as a blank template to run experiments.
# The sys.argv = [demo_infra_usage.py (time_stamped_folder) (integer)]
# so use the (time_stamped_folder)/res/ to save outputs
# and use the (integer) to define experiment settings from a lookup table.
#
# To run this script 10 times distributed over CSC machines, on local computer type:
#  $ python fork0_to_csc.py \$HOME/cond_bayes_opt/scripts/demo_infra_usage.py 10 -v
#
# see fork0_to_csc.py for further help.


def run(args):
    """
    This is a stupid function just for demonstration purposes.
    It takes the args and prints something and saves something.
    In general any python experiment running code can go here.
    As long as the conda env and git branch are correctly set.
    """

    # define a list of all job settings (here is some random crap)
    np.random.seed(1)
    all_job_settings = np.random.uniform(size=(1000,))

    # use the args.k as a lookup into all_job_settings
    this_job_setting = all_job_settings[args.k]
    this_job_savefile = args.dirname + "/res/" + str(args.k)

    # Now to run some code!
    # Let's print something, say the conda env, args, computer and job setting?
    # get current conda environment that called this script
    conda_env = os.environ["CONDA_DEFAULT_ENV"]

    # get current computer name
    hostname = sp.check_output(["hostname"], shell=True).decode()[:-1]

    # IMPORT AND RUN MODULES
    import experiment_manager
    from forking_CSC.fork0_to_csc import U

    number_of_csc_machines = len(U)
    # print(number_of_csc_machines)
    seed = 0
    for _ in range(33):
        experiment_names = [
            "ONESHOTHYBRIDKG_GP_synthetic_3_dim2_l0.1",
            "ONESHOTHYBRIDKG_GP_synthetic_3_dim2_l0.4",
            "ONESHOTHYBRIDKG_GP_synthetic_3_dim4_l0.1",
            "ONESHOTHYBRIDKG_GP_synthetic_3_dim4_l0.4",
            # "ONESHOTKG_GP_synthetic_10_dim2_l0.1",
            # "ONESHOTKG_GP_synthetic_10_dim2_l0.4",
            # "ONESHOTKG_GP_synthetic_10_dim4_l0.1",
            # "ONESHOTKG_GP_synthetic_10_dim4_l0.4",

                            # "MCKG_GP_synthetic_10_dim2_l0.1",
                            # "MCKG_GP_synthetic_10_dim2_l0.4",
                            #                 "MCKG_GP_synthetic_3_dim2_l1",
                            # "MCKG_GP_synthetic_10_dim4_l0.1",
                            # "MCKG_GP_synthetic_10_dim4_l0.4",

            #                 "MCKG_GP_synthetic_3_dim2_l0.1",
            #                 "MCKG_GP_synthetic_3_dim2_l0.4",
            # #                 "MCKG_GP_synthetic_3_dim2_l1",
            #                 "MCKG_GP_synthetic_3_dim4_l0.1",
            #                 "MCKG_GP_synthetic_3_dim4_l0.4",
            #                 "MCKG_GP_synthetic_3_dim4_l1",
            #                 "MCKG_GP_synthetic_3_dim6_l0.1",
            #                 "MCKG_GP_synthetic_3_dim6_l0.4",
            #                 "MCKG_GP_synthetic_3_dim6_l1",
            #                 "DISCKG_GP_synthetic_1000_dim2_l0.1",
            #                 "DISCKG_GP_synthetic_1000_dim2_l0.4",
                            # "DISCKG_GP_synthetic_1000_dim2_l1",
                            # "DISCKG_GP_synthetic_1000_dim4_l0.1",
                            # "DISCKG_GP_synthetic_1000_dim4_l0.4",
                            # "DISCKG_GP_synthetic_1000_dim4_l1",
                            # "DISCKG_GP_synthetic_1000_dim6_l0.1",
                            # "DISCKG_GP_synthetic_1000_dim6_l0.4",
                            # "DISCKG_GP_synthetic_1000_dim6_l1",
                            # "DISCKG_GP_synthetic_dim2_l0.1",
                            # "DISCKG_GP_synthetic_dim2_l0.4",
                            # "DISCKG_GP_synthetic_dim2_l1",
                            # "DISCKG_GP_synthetic_dim4_l0.1",
                            # "DISCKG_GP_synthetic_dim4_l0.4",
                            # "DISCKG_GP_synthetic_dim4_l1",
                            # "DISCKG_GP_synthetic_dim6_l0.1",
                            # "DISCKG_GP_synthetic_dim6_l0.4",
                            # "DISCKG_GP_synthetic_dim6_l1",
                            #"DISCKG_Branin_1000",
                            # "DISCKG_Rosenbrock_2",
                            # "DISCKG_Rosenbrock_1000",
                            # "DISCKG_Hartmann_2",
                            # "DISCKG_Hartmann_1000",
                            # "HYBRIDKG_Branin_2",
                            # "HYBRIDKG_Branin_10",
                            # "HYBRIDKG_Rosenbrock_2",
                            # "HYBRIDKG_Rosenbrock_10",
                            # "HYBRIDKG_Hartmann_2",
                            # "HYBRIDKG_Hartmann_10",
                            # "MCKG_Branin_2",
                            # "MCKG_Branin_10",
                            # "RANDOMKG_Branin",
                            # "ONESHOTKG_Branin_2",
                            # "ONESHOTKG_Branin_10",
                            # "ONESHOTKG_Branin_125"
                            # "MCKG_Rosenbrock_2",
                            # "MCKG_Rosenbrock_10",
                            # "MCKG_Hartmann_2",
                            # "MCKG_Hartmann_10",
                            ]
        # experiment_names = ["DISCKG_Hartmann_2"]
        for exp_name in experiment_names:
            # print("args.k + seed",args.k + seed)
            if args.k + seed>100:
                raise

            experiment_manager.main(exp_names=exp_name, seed=args.k + seed)
            # print(args.k + seed, exp_name)
        seed += number_of_csc_machines


    # experiment_manager(args.k)

    # save something to hard drive in /res/ subfolder
    with open(this_job_savefile, "w") as f:
        f.write(output + "\n\n")

    # end of demo
    print("\nOutput saved to file: ", this_job_savefile, "\n\n\n\n")


def callbash(cmd, silent=False):
    if silent:
        _ = sp.check_output(["/bin/bash", "-c", cmd])
    else:
        _ = sp.call(["/bin/bash", "-c", cmd], stderr=sp.STDOUT)


if __name__ == "__main__":
    ####################################### WARNING ####################################
    # ALL EXPERIMENT RUNNERS MUST HAVE THE FOLLOWING ARGS!!!! DO NOT CHANGE THIS!!!!
    ####################################### WARNING ####################################

    parser = argparse.ArgumentParser(
        description="Run k-th experiment from the look-up table"
    )
    parser.add_argument("dirname", type=str, help="Experiment directory")
    parser.add_argument(
        "k", type=int, help="Row in look-up table corresponding to specific experiment"
    )

    # These arguments are assumed by the forking files. Use args.dirname+"/res/" as a results output directory.
    # In this file, define a list of aaallll the experiments you want to run and use args.k as a lookup index
    # within the list. Save the output as args.dirname+"/res/" + str(args.k) (the /res/ folder has been made already)

    args = parser.parse_args()
    run(args)
