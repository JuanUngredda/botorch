import torch
import os
import pickle as pkl


class save_results():

    def __init__(self, kg_type, keys, bounds, seed, save_folder):
        self.seed = seed
        self.keys = keys
        self.bounds = bounds
        self.kg_type = kg_type
        self.x_best_observed = []
        self.y_best_observed = []
        self.x_train = None
        self.y_train = None
        self.model_hs = []
        self.save_path = save_folder

    def save(self, model, x_train, y_train):
        self.x_best_observed.append(x_train[torch.argmax(y_train)])
        self.y_best_observed.append(torch.max(y_train))
        self.x_train = x_train
        self.y_train = y_train
        self.model_hs.append(model.covar_module.base_kernel.lengthscale.detach().squeeze())

        output = {"KG_type": self.kg_type,
                  "seed": self.seed,
                  "columns": self.keys,
                  "bounds": self.bounds,
                  "best_observed_x": self.x_best_observed,
                  "best_observed_kg": self.y_best_observed,
                  "X_train": self.x_train,
                  "Y_train": self.y_train,
                  "model_hs": self.model_hs}

        if os.path.isdir(self.save_path) == False:
            os.makedirs(self.save_path, exist_ok=True)

        with open(self.save_path + "/" + str(self.seed) + ".pkl", "wb") as f:
            pkl.dump(output, f)
