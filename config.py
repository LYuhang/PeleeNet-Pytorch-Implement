# -*- coding: utf-8 -*-

import os
import torch

class Config():
    def __init__(self):
        # general param
        self.RETRAIN = True
        self.USE_CUDA = False #torch.cuda.is_available()

        # define the data paths
        self.RAW_TRAIN_DATA = "./data/cards_250_7/cards_for_train"
        self.RAW_TEST_DATA = "./data/cards_250_7/cards_for_eval"
        # define the source path
        self.SOURCE_DIR_PATH = {
            "MODEL_DIR" : "./source/models/"
        }
        # define the file path
        self.LABEL_TO_NAME_PATH = "./source/label_to_name_dict.pkl"
        self.NAME_TO_LABEL_PATH = "./source/name_to_label_dict.pkl"

        # check the path
        self.check_dir()

        # define the param of the picture
        self.WIDTH = 488
        self.HEIGHT = 488
        self.CHANNEL = 3
        self.NUM_CLASS = 250
        self.BATCH_SIZE = 30
        self.NUM_EPOCHS = 500
        self.LEARNING_RATE = 0.001

    def check_dir(self):
        '''
        This function is used to check the dirs.if data path
        does not exists, raise error.if source path does not
        exits, make new dirs.
        :return: None
        '''
        # check the data path
        if not os.path.exists(self.RAW_TEST_DATA):
            raise Exception("==> Error: Data path %s does not exist." % self.RAW_TEST_DATA)
        if not os.path.exists(self.RAW_TRAIN_DATA):
            raise Exception("==> Error: Data path %s does not exist." % self.RAW_TRAIN_DATA)

        # check the source path
        for name, path in self.SOURCE_DIR_PATH.items():
            if not os.path.exists(path):
                print("==> Creating %s : %s" % (name, path))
                os.makedirs(path)