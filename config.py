"""Configurations & parameters."""

import platform

class Config:

    dataset = 'coco'

    datapath = ''

    batch_size = 128

    num_epochs = 100

    learning_rate = 1e-3

    coco_dir = './data/images'

    input_json = './data/cocotalk.json'

opt = Config()