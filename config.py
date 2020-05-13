import json


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as fjson:
            conf = json.load(fjson)
        self.dim = conf["dim"]
        self.alpha = conf["alpha"]
        self.beta = conf["beta"]
        self.theta = conf["theta"]
        self.delta = conf["delta"]
        self.gamma = conf["gamma"]
        self.learning_rate = conf["learning_rate"]
        self.batch_size = conf["batch_size"]
        self.neg_size = conf["neg_size"]
        self.test_batch_size = conf["test_batch_size"]
        self.regularization = conf["regularization"]
        self.data_path = conf["data_path"]
        self.save_path = conf["save_path"]
        self.cuda = conf["cuda"]
        self.max_step = conf["max_step"]
        self.warm_up_step = conf["warm_up_step"]
        self.log_step = conf["log_step"]
        self.test_log_step = conf["test_log_step"]
        self.update_step = conf["update_step"]
        self.init_checkpoint = conf["init_checkpoint"]
        self.use_old_optimizer = conf["use_old_optimizer"]
        self.cpu_num = conf["cpu_num"]


config = Config("./config/config.json")
