# -*- coding: utf-8 -*-

import os
import pickle
from config import Config

class preprocess():
    def __init__(self, config):
        self.config = config

    def gen_label_dict(self):
        # Get the training data path
        train_path = self.config.RAW_TRAIN_DATA
        # Get the dirname in the train data path
        name_to_label_dict = dict()
        n = 0
        print("==> Generating name_to_label_dict.")
        for name in os.listdir(train_path):
            ph = os.path.join(train_path, name)
            if os.path.isdir(ph):
                label = name.split("_")[0] # Get the label
                name_to_label_dict[label] = n
                n += 1
        pickle.dump(name_to_label_dict, open(self.config.NAME_TO_LABEL_PATH, "wb"))
        print("There are %d labels" % len(name_to_label_dict))

        print("==> Generating label_to_name_dict.")
        label_to_name_dict = {l:n for n,l in name_to_label_dict.items()}
        pickle.dump(label_to_name_dict, open(self.config.LABEL_TO_NAME_PATH, "wb"))

if __name__ == "__main__":
    conf = Config()
    pre = preprocess(conf)
    if not os.path.exists(conf.LABEL_TO_NAME_PATH) or not os.path.exists(conf.NAME_TO_LABEL_PATH):
        print("==> Generating label dict.")
        pre.gen_label_dict()