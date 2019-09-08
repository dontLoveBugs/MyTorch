# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:05
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : validator.py.py
"""

import torch
import numpy as np

from easydict import EasyDict as edict


from modules.utils.img_utils import normalize
from modules.utils.visualize import get_color_pallete
from modules.metircs.seg.metric import SegMetric


class Validator(object):
    def __init__(self, dataset, config, device, out_id=[1]):
        """
        :param model: module
        :param device: validator should be run in device(local rank = 0).
        :param config: easydict
        :param out_id: [int, ...],
                        the index of eval images which you wanna visualize.
        """
        assert isinstance(config, edict), "config is not easy dict."

        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = config.data.num_classes
        self.image_mean = config.data.image_mean
        self.image_std = config.data.image_std

        self.config = config

        self.device = device
        self.metric = SegMetric(n_classes=self.class_num)
        self.out_id = out_id

        self.val_func = None

    def eval(self, model):
        self.val_func = model
        self.val_func.eval()
        out_images = None
        sum_loss = 0.0
        for idx in range(self.ndata):
            data = self.dataset[idx]['data']
            label = self.dataset[idx]['label']
            loss, pred = self.val_func_process(data, label, self.device)
            sum_loss += loss.item()
            self.metric.update(pred, label)

            if idx in self.out_id:
                tmp_imgs = np.vstack([data,
                                      get_color_pallete(data, label,
                                                        self.dataset.get_class_colors(), self.config.data.background),
                                      get_color_pallete(data, pred,
                                                        self.dataset.get_class_colors(), self.config.data.background)])
                out_images = tmp_imgs if out_images is None else np.hstack(out_images, tmp_imgs)

        return sum_loss / self.dataset.get_length(), self.metric.get_scores(), out_images

    def val_func_process(self, data, label, device=None):
        data = self.pre_process(data)
        data = torch.FloatTensor(data).cuda(device).unsqueeze(0)
        label = torch.LongTensor(label).cuda(device).unsqueeze(0)

        with torch.no_grad():
            loss, pred = self.val_func(data, label)
            pred = torch.exp(pred)  # the output of network is log_softmax,

        return pred.unsqueeze(0), loss.item()

    def pre_process(self, img):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)
        p_img = p_img.transpose(2, 0, 1)
        return p_img
