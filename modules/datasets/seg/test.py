#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-09-12 14:23
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : test.py
"""


import cv2
import numpy as np
from PIL import Image

import time
# BGR

end = time.time()
img_BGR = cv2.imread('left.png', cv2.IMREAD_COLOR)
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
a = np.array(img_RGB, dtype=None)
print('opencv cost time:',time.time() - end)

end = time.time()
# RGB
b = np.array(Image.open('left.png', mode='r').convert('RGB'), dtype=None)
print('pillow cost time:', time.time()-end)

from modules.utils.img_utils import pad_image_to_shape

crop_size = 512

a_crop, margin = pad_image_to_shape(a, crop_size, cv2.BORDER_CONSTANT, 0)
print(margin)

cv2.imshow('a crop', a_crop)
cv2.imwrite('acrop.png', a_crop)
# cv2.waitKey()