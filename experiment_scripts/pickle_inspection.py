import pickle

# read python dict back from the file
pkl_file = open(
    "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/Branin/MCKG/0.pkl",
    "rb",
)
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print(mydict2["method_times"])

pkl_file = open(
    "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/Branin/HYBRIDKG/0.pkl",
    "rb",
)
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print(mydict2["method_times"])
