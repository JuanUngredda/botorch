import pickle

# read python dict back from the file

pkl_file = open("./results/DISCKG/0.pkl", "rb")
dict = pickle.load(pkl_file)
pkl_file.close()
print(dict)
