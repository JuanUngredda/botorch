class parameter_setter():
    def __init__(self, experiment_name) -> None:
        self.config_number_raw_samples_acq_opt = None
        self.experiment_name = experiment_name
        self.config_number_fantasies = None
        self.config_number_discrete_points = None
        self.config_number_restarts_inner_opt = 5 # TODO: Change. This is only for testing purposes
        self.config_number_raw_samples_inner_opt = None
        self.config_number_restarts_acq_opt = 5 #TODO: This is only for testing purposes

    def build(self, configuration_dictionary, labels):
        for i_label, label in enumerate(labels):
            if label == "num_fantasies":
                self.config_number_fantasies = int(configuration_dictionary["num_fantasies"])
            elif label == "num_discrete_points":
                self.config_number_discrete_points = int(configuration_dictionary["num_discrete_points"])
            elif label == "raw_samples_inner_optimizer":
                self.config_number_raw_samples_inner_opt = int(configuration_dictionary["raw_samples_inner_optimizer"])
            elif label == "raw_samples_acq_optimizer":
                self.config_number_raw_samples_acq_opt = int(configuration_dictionary["raw_samples_acq_optimizer"])
            else:
                print("considered label is not in the dictionary")

    def get_number_fantasies(self):
        return self.config_number_fantasies

    def get_number_discrete_points(self):
        return self.config_number_discrete_points

    def get_number_restarts_inner_opt(self):
        return self.config_number_restarts_inner_opt

    def get_number_raw_samples_inner_opt(self):
        return self.config_number_raw_samples_inner_opt

    def get_number_restarts_acq_opt(self):
        return self.config_number_restarts_acq_opt

    def get_number_raw_samples_acq_opt(self):
        return self.config_number_raw_samples_acq_opt
