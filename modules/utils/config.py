# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/4 20:03
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import os
import os.path as osp
import json
import numpy as np
import time
from easydict import EasyDict as edict

from modules.utils.pyt_utils import parse_frac_str


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

    config['image_mean'] = np.array(config['image_mean'])  # 0.485, 0.456, 0.406
    config['image_std'] = np.array(config['image_std'])
    return config


def save_config(config_file, config):
    with open(config_file, "w") as dump_f:
        json.dump(config, dump_f, indent=4)


class Config(object):
    """
    config: json ---> edict
    """

    def __init__(self, config_file=None):
        assert os.path.exists(config_file), 'config file is not existed.'
        self.config_file = config_file
        self.load()
        self.create_log()

    def load(self, defaults=None):
        with open(self.config_file, "r") as fd:
            self.config_json = json.load(fd)

        # if defaults is not None:
        #     _merge(defaults, self.config)

        self.config = edict(self.config_json)
        self.config.data.image_mean = np.array(self.config.data.image_mean)
        self.config.data.image_std = np.array(self.config.data.image_std)

        # eval stride_rate
        if self.config.get('eval').get('stride_rate') is not None:
            self.config.eval.stride_rate = \
                parse_frac_str(self.config.get('eval').get('stride_rate'))

    def save(self, filename):
        # assert osp.exists(self.config_json), "config json is not existed."
        with open(filename, "w") as dump_f:
            json.dump(self.config_json, dump_f, indent=4)

    def get_config(self):
        return self.config

    def create_log(self):
        self.config.log = edict()

        # snapshot dir setting
        self.config.log.abs_dir = osp.realpath(".")
        self.config.log.this_dir = self.config.log.abs_dir.split(osp.sep)[-1]
        self.config.log.root_dir = self.config.log.abs_dir[
                                   :self.config.log.abs_dir.index(self.config.environ.repo_name) +
                                    len(self.config.environ.repo_name)]
        self.config.log.log_dir = osp.abspath(osp.join(self.config.log.root_dir, 'log', self.config.log.this_dir))
        self.config.log.log_dir_link = osp.join(self.config.log.abs_dir, 'log')

        snapshot_dir = osp.abspath(osp.join(self.config.log.log_dir, self.config.snapshot.name))

        # a new training process, but not set a new snapshot name to record the new training process.
        if osp.exists(snapshot_dir) and \
                not osp.exists(self.config.model.continue_path):
            snapshot_dir += '_new'

        self.config.log.snapshot_dir = snapshot_dir

        # log file
        exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self.config.log.log_file = self.config.log.log_dir + '/log_' + exp_time + '.log'
        self.config.log.link_log_file = self.config.log.log_file + '/log_last.log'
        self.config.log.val_log_file = self.config.log.log_dir + '/val_' + exp_time + '.log'
        self.config.log.link_val_log_file = self.config.log.log_dir + '/val_last.log'

    def __str__(self):
        str = ''
        for k, v in enumerate(self.config):
            str += str(v) + '\n'
            print(k, v)
        return str
