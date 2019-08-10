# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/10 21:11
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : test.py
"""

from easydict import EasyDict as edit

x = edit()

x.a = 1/3

print(x)


json = {'a':1/3}
print(json)

# 测试json文件
from modules.utils.config import Config

config = Config('./config.json').get_config()
# print(config)

from fractions import Fraction
stride_rate = config.eval.stride_rate
stride_rate = float(Fraction(stride_rate))
print(stride_rate)
