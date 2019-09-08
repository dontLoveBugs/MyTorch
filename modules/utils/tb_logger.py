# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:51
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : tb_logger.py
"""


import os
import shutil
import socket
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from modules.utils.pyt_utils import ensure_dir


def create_logger(logdir=None):
    # assert os.path.exists(logdir), 'Log file dir is not existed.'
    ensure_dir(logdir)

    log_path = os.path.join(logdir, 'tensorboard',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)
    return logger


class Logger(object):

    def __init__(self, logdir=None):
        super(Logger, self).__init__()
        self.logger = create_logger(logdir)

    def add_scalar_dict(self, tag, scalar_dict, it):
        """
        :param tag: str, train, eval or test.
        :param scalar_dict:
                type: dict
                {'scalar name', scalar value, ...}
        :param it: global step
        """

        assert isinstance(scalar_dict, dict), 'scalar dict must be dict type.'
        for k, v in scalar_dict.items():
            self.logger.add_scalar(tag + '/' + k, v, it)

    def add_scalar_dict_list(self, tag, scalar_dict_list, it):
        """
        :param tag: str, it generally is 'trainval'
        :param scalars_list:
                type: list
                [{'scalar name': scalar value, ...}, ]
        :param it: global step
        """

        assert isinstance(scalar_dict_list, list), 'scalars list must be list type.'
        for k, v in enumerate(scalar_dict_list):
            self.logger.add_scalars(tag, v, it)

    def close(self):
        self.logger.close()
