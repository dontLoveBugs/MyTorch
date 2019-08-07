# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:53
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : test.py
"""


class Engine(object):
    def __init__(self):
        self.im = 'test'

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if type is not None:
            self.logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False

        print('finished.')


with Engine() as engine:
    print(engine.im)
    raise NotImplementedError


