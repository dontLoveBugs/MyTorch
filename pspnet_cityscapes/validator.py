# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:05
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : validator.py.py
"""
import sys

import cv2
import torch
import numpy as np

import torch.nn.functional as F
from easydict import EasyDict as edict

from tqdm import tqdm

from modules.utils.img_utils import normalize, pad_image_to_shape
from modules.utils.visualize import get_color_pallete
from modules.metircs.seg.metric import SegMetric


class Validator(object):
    def __init__(self, dataset, config, device, ignore_index=-1, out_id=[1]):
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
        self.crop_size = config.eval.crop_size

        self.config = config

        self.ignore_index = ignore_index
        self.device = device
        self.metric = SegMetric(n_classes=self.class_num)
        self.out_id = out_id

        self.val_func = None

    def run(self, model):
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
            # print('#val:', data.shape, label.shape)

            pred, loss = self.whole_eval(data, label,
                                         input_size=self.crop_size,
                                         output_size=data.shape[:2],
                                         device=self.device)
            sum_loss += loss
            self.metric.update(pred, label)

            if idx in self.out_id:
                tmp_imgs = np.vstack([data,
                                      get_color_pallete(data, label,
                                                        self.dataset.get_class_colors(), self.config.data.background),
                                      get_color_pallete(data, pred,
                                                        self.dataset.get_class_colors(), self.config.data.background)])
                # out_images = tmp_imgs if out_images is None else np.hstack([out_images, tmp_imgs])
                if out_images is None:
                    out_images = tmp_imgs
                else:
                    new_height = out_images.shape[0]
                    # new_width = int(out_images.shape[1] / new_height * tmp_imgs.shape[1])
                    new_width = int(tmp_imgs.shape[1] / tmp_imgs.shape[0] * new_height)
                    # tmp_imgs.resize((new_height, new_width, tmp_imgs.shape[2]))
                    tmp_imgs = cv2.resize(tmp_imgs, (new_width, new_height),
                                          interpolation=cv2.INTER_LINEAR)
                    # print(out_images.shape, tmp_imgs.shape)
                    out_images = np.hstack([out_images, tmp_imgs])

            print_str = 'Validation {}/{}:'.format(idx + 1, self.ndata)
            pbar.set_description(print_str, refresh=False)

        # empty the eval cuda cache.
        torch.cuda.empty_cache()
        return sum_loss / self.dataset.get_length(), self.metric.get_scores(), out_images

    # evaluate the whole image at once
    def whole_eval(self, img, label, output_size, input_size=None, device=None):
        if input_size is not None:
            img, label, margin = self.process_image(img, label, input_size)
        else:
            img, label = self.process_image(img, label, input_size)

        pred, loss = self.val_func_process(img, label, device)

        # print('pred = ', pred.shape)
        # print('margin = ', margin)
        if input_size is not None:
            pred = pred[:, margin[0]:(pred.shape[1] - margin[1]),
                   margin[2]:(pred.shape[2] - margin[3])]

        pred = pred.transpose(1, 2, 0)

        if output_size is not None:
            pred = cv2.resize(pred,
                              (output_size[1], output_size[0]),
                              interpolation=cv2.INTER_LINEAR)

        return pred, loss

    def to_tensor(self, img, gt, device):
        img = torch.FloatTensor(img).cuda(device).unsqueeze(0)
        gt = torch.LongTensor(gt).cuda(device).unsqueeze(0)

        return img, gt

    def val_func_process(self, data, label, device=None):
        data, label = self.to_tensor(data, label, device)

        with torch.no_grad():
            loss, pred = self.val_func(data, label)
            pred = torch.exp(pred)  # the output of network is log_softmax,
            # print('# pred ', pred.shape)
            pred = pred.argmax(1)

        return pred.cpu().numpy(), loss.item()

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

    def process_image(self, img, gt, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_gt, margin = pad_image_to_shape(gt, crop_size,
                                              cv2.BORDER_CONSTANT, value=self.ignore_index)

            p_img = p_img.transpose(2, 0, 1)

            return p_img, p_gt, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img, gt