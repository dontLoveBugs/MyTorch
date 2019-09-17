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

from modules.utils.img_utils import normalize, pad_image_to_shape, center_crop
from modules.utils.visualize import get_color_pallete
from modules.metircs.seg.metric import SegMetric


class Validator(object):
    def __init__(self, dataset, config, device, ignore_index=-1, out_fn=[1]):
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
        self.stride_rate = config.eval.stride_rate
        self.multi_scales = config.eval.scale_array

        self.config = config

        self.ignore_index = ignore_index
        self.device = device
        self.metric = SegMetric(n_classes=self.class_num)
        self.out_fn = out_fn

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

            data = center_crop(data, self.crop_size)
            label = center_crop(label, self.crop_size)

            # pred, loss = self.sliding_eval(data, label,
            #                              crop_size=self.crop_size,
            #                              stride_rate=self.stride_rate,
            #                              device=self.device)
            pred, loss = self.whole_eval(data, label, device=self.device)
            sum_loss += loss
            self.metric.update(pred, label)

            if self.dataset[idx]['fn'] in self.out_fn:
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

            print_str = 'GPU{}, Validation {}/{}:'.format(self.device, idx + 1, self.ndata)
            pbar.set_description(print_str, refresh=False)

        # empty the eval cuda cache.
        torch.cuda.empty_cache()
        return sum_loss / self.dataset.get_length(), self.metric.get_scores(), out_images

    # evaluate the whole image at once
    def whole_eval(self, img, label, output_size=None, input_size=None, device=None):
        if input_size is not None:
            img, label, margin = self.process_image(img, label, input_size)
        else:
            img, label = self.process_image(img, label, input_size)

        pred, loss = self.val_func_process(img, label, device)
        pred = pred.argmax(dim=2)
        pred = pred.cpu().numpy()

        if input_size is not None:
            pred = pred[:, margin[0]:(pred.shape[1] - margin[1]),
                   margin[2]:(pred.shape[2] - margin[3])]
        pred = pred.transpose(1, 2, 0)

        if output_size is not None:
            pred = cv2.resize(pred,
                              (output_size[1], output_size[0]),
                              interpolation=cv2.INTER_LINEAR)

        return pred, loss

    # slide the window to evaluate the image
    def sliding_eval(self, img, label, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        scale_pred = np.zeros((ori_rows, ori_cols, self.class_num))
        scale_loss = 0.0

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            label_scale = cv2.resize(label, None, fx=s, fy=s,
                                     interpolation=cv2.INTER_LINEAR)

            new_rows, new_cols, _ = img_scale.shape
            tmp_pred, tmp_loss = self.scale_process(img_scale, label_scale,
                                                    (ori_rows, ori_cols),
                                                    crop_size, stride_rate, device)
            scale_pred += tmp_pred
            scale_loss += tmp_loss

        pred = scale_pred / len(self.multi_scales)
        pred = pred.argmax(2)
        loss = scale_loss / len(self.multi_scales)
        return pred, loss

    def scale_process(self, img, label, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, label_data, margin = self.process_image(img, label, crop_size)
            score, loss = self.val_func_process(input_data, label_data, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)
            label_pad, _ = pad_image_to_shape(label, crop_size,
                                              cv2.BORDER_CONSTANT, value=self.ignore_index)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)
            data_loss = 0.0
            count_loss = 0
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    label_sub = label_pad[s_y:e_y, s_x: e_x]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, label_data, tmargin = self.process_image(img_sub, label_sub, crop_size)
                    temp_score, tmp_loss = self.val_func_process(input_data, label_data, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
                    data_loss += tmp_loss
                    count_loss += 1.0

            loss = data_loss / count_loss
            score = data_scale / count_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        # score = score.argmax(dim=2)
        score = cv2.resize(score.cpu().numpy(),
                           (ori_shape[1], ori_shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        return score, loss

    def to_tensor(self, img, gt, device):
        img = torch.FloatTensor(img).cuda(device, non_blocking=True).unsqueeze(0)
        gt = torch.LongTensor(gt).cuda(device, non_blocking=True).unsqueeze(0)

        return img, gt

    def val_func_process(self, data, label, device=None):
        data, label = self.to_tensor(data, label, device)

        with torch.no_grad():
            loss, pred = self.val_func(data, label)
            pred = torch.exp(pred)  # the output of network is log_softmax,

        return pred[0], loss.item()

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