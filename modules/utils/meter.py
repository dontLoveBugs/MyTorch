# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:46
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : meter.py
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