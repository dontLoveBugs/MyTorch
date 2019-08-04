# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/4 20:03
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


import json
import numpy as np


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=None):
    with open(config_file, "r") as fd:
        config = json.load(fd)

    if defaults is not None:
        _merge(defaults, config)

    config['image_mean'] = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
    config['image_std'] = np.array([0.229, 0.224, 0.225])
    return config


def save_config(config_file, config):
    with open(config_file, "w") as dump_f:
        json.dump(config, dump_f, indent=4)