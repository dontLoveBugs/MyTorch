# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:05
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : validator.py.py
"""
import sys

import torch
import numpy as np

import torch.nn.functional as F
from easydict import EasyDict as edict

from tqdm import tqdm

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
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(self.ndata), file=sys.stdout,
                    bar_format=bar_format)

        self.val_func = model
        self.val_func.eval()
        out_images = None
        sum_loss = 0.0
        for idx in pbar:
            data = self.dataset[idx]['data']
            label = self.dataset[idx]['label']
            label -= 1

            # print('#val:', data.shape, label.shape)

            loss, pred = self.val_func_process(data, label, self.device)
            sum_loss += loss

            pred, label = self.post_process(pred, label)
            self.metric.update(pred, label)

            if idx in self.out_id:
                tmp_imgs = np.vstack([data,
                                      get_color_pallete(data, label,
                                                        self.dataset.get_class_colors(), self.config.data.background),
                                      get_color_pallete(data, pred,
                                                        self.dataset.get_class_colors(), self.config.data.background)])
                out_images = tmp_imgs if out_images is None else np.hstack(out_images, tmp_imgs)

            print_str = 'Validation {}/{}:'.format(idx + 1, self.ndata)
            pbar.set_description(print_str, refresh=False)

        return sum_loss / self.dataset.get_length(), self.metric.get_scores(), out_images

    def val_func_process(self, data, label, device=None):
        data, label = self.pre_process(data, label)
        data = torch.FloatTensor(data).cuda(device).unsqueeze(0)
        label = torch.LongTensor(label).cuda(device).unsqueeze(0)

        with torch.no_grad():
            loss, pred = self.val_func(data, label)
            pred = torch.exp(pred)  # the output of network is log_softmax,
            pred = pred.argmax(1)

        return loss.item(), pred.squeeze(0).cpu().numpy()

    def pre_process(self, img, gt=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)
        p_img = p_img.transpose(2, 0, 1)

        if gt is not None:
            # p_gt = gt - 1
            p_gt = gt
            return p_img, p_gt

        return p_img

    def post_process(self, pred, gt):
        if pred.shape != gt.shape:
            pred = F.interpolate(pred, size=gt.size()[-2:], mode='bilinear', align_corners=True)

        return pred, gt