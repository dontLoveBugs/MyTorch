# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:46
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : average_meter.py
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictAverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.key_list = None

    def reset(self):
        self.val = {key:0. for key in self.key_list}
        self.avg = {key:0. for key in self.key_list}
        self.sum = {key:0. for key in self.key_list}
        self.count = {key:0 for key in self.key_list}

    def update(self, val_dict, n_dict=None):
        if self.key_list is None:
            self.key_list = val_dict.keys()
            self.reset()

        if isinstance(n_dict, (int, float)):
            new_n_dict = {k: n_dict for k in val_dict.keys()}
            n_dict = new_n_dict

        self.val = val_dict
        for k in val_dict.keys():
            self.sum[k] += val_dict[k] * n_dict[k]
            self.count[k] += n_dict[k]
            self.avg[k] = self.sum[k] / self.count[k]

    def info(self):
        str = '{'
        for k, v in self.avg.items():
            str += '{}: {:.4f}, '.format(k, v)

        return str.rstrip(', ') + '}'