import pickle

# read python dict back from the file
pkl_file = open(
    "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/TEST1/MCKG/0.pkl",
    "rb",
)
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print(len(mydict2["x"]))
print(mydict2)
