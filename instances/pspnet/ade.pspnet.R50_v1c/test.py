# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/5 20:42
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


from easydict import EasyDict as edict

a = edict()
a.aa = edict()
a.aa.aaa = 1
a.aa.bbb = 2
a.bb = edict()
a.bb.aaa = 3
a.bb.bbb = 4
a.bb.ccc = 5

print(a)

for k, v in a:
    print(k, v)

str0 = ''
for k, v in a:
    if isinstance(v, edict):
        str0 += k + ':\n'
        for kk, vv in v:
            str0 += '  ' + kk + ': ' + str(vv) + '\n'
    else:
        str0 += k + ': ' + str(v) + '\n'

print(str0)