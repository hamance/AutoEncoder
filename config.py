#encoding:utf-8
"""Configurations & parameters."""
import os
import platform
import time
import warnings


class Config:

    dataset = 'coco'

    datapath = ''

    batch_size = 128

    num_epochs = 100

    learning_rate = 1e-3

    coco_dir = 'g:\\image_caption\\zips\\coco\\2014'

    input_json = 'g:\\image_caption\\coco\\lrt\\cocotalk.json'

    def parse(self, kwargs):
            u"""根据字典kwargs 更新 config参数."""
            for key, val in kwargs.items():
                if not hasattr(self, key):
                    warnings.warn("Warning: opt has not attribut %s" % key)
                setattr(self, key, val)

    def show(self):
        """Print configs."""
        print('user config:')
        params = {}
        params.update(self.__class__.__dict__)
        # When assigning to opt, self.__dict__ is updated
        # Thus self.__dict__ stores the newer value
        params.update(self.__dict__)
        for k in sorted(params.keys()):
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = Config()
