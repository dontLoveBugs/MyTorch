# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:53
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : test.py
"""


# class Engine(object):
#     def __init__(self):
#         self.im = 'test'
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, type, value, tb):
#         if type is not None:
#             self.logger.warning(
#                 "A exception occurred during Engine initialization, "
#                 "give up running process")
#             return False
#
#         print('finished.')
#
#
# with Engine() as engine:
#     print(engine.im)
#     raise NotImplementedError



from modules.utils.average_meter import DictAverageMeter
from easydict import EasyDict as edcit

ed0 = edcit()
ed0.a = 5
ed0.b = 10

ed1 = edcit()
ed1.a = 3
ed1.b = 2

if isinstance(ed0, dict):
    print('edict is dict')

avg_meter = DictAverageMeter()

avg_meter.update(ed0, n_dict=1)
avg_meter.update(ed1, n_dict=1)

print(avg_meter.info())

res = avg_meter.avg
print(avg_meter.avg)
print(avg_meter.sum)

from modules.datasets.seg.ade import ADE

ade = ADE(root='/data/wangxin/ADEChallengeData2016', split='train')
print(ade.get_class_colors())

import torch

a = torch.randn(4, 3, 225, 225)
b = torch.randn(4, 1, 225, 225)

for a0, b0 in zip(a, b):
    # print(a0.shape, b0.shape)
    print(a0.flatten().shape, b0.flatten().shape)

