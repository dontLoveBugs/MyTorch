# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/7 22:05
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : validator.py.py
"""
import time

import cv2
import torch
import numpy as np

import torch.multiprocessing as mp
from easydict import EasyDict as edict
from tqdm import tqdm

from modules.engine.logger import get_logger
from modules.utils.img_utils import normalize, pad_image_to_shape
from modules.utils.visualize import get_color_pallete
from modules.metircs.seg.metric import hist_info

logger = get_logger()


class Validator(object):
    def __init__(self, dataset, config, devices=[0, 1, 2, 3], ignore_index=-1, out_id=[1]):
        """
        :param model: module
        :param devices: validator should be run in devices([0, 1, 2, 3]).
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

        self.multi_scales = config.eval.scale_array
        self.crop_size = config.eval.crop_size
        self.is_flip = config.eval.flip

        self.ignore_index = ignore_index

        self.config = config

        self.devices = devices
        self.out_id = out_id

        self.val_func = None

        self.context = mp.get_context('spawn')
        self.results_queue = self.context.Queue(self.ndata)

    def to_tensor(self, img, gt, device):
        img = torch.FloatTensor(img).cuda(device).unsqueeze(0)
        gt = torch.LongTensor(gt).cuda(device).unsqueeze(0)

        return img, gt

    def val_func_process(self, data, label, device=None):
        data, label = self.to_tensor(data, label, device)
        self.val_func.to(data.get_device())

        with torch.no_grad():
            loss, pred = self.val_func(data, label)
            pred = pred[0]

            if self.is_flip:
                data = data.flip(-1)
                loss_flip, pred_flip = self.val_func(data)
                pred_flip = pred_flip[0]
                pred = 0.5 * (pred + pred_flip.flip(-1))
                loss = 0.5 * (loss + loss_flip)

            pred = torch.exp(pred)  # the output of network is log_softmax,

        return pred, loss

    def worker(self, model, shred_list, device):
        self.val_func = model
        self.val_func.eval()

        for idx in shred_list:
            data = self.dataset[idx]['data']
            label = self.dataset[idx]['label']
            pred, loss = self.whole_eval(data, label,
                                         input_size=self.crop_size,
                                         output_size=data.shape[:2],
                                         device=device)

            tmp_hist, _, _ = hist_info(self.class_num, pred, label)

            if idx in self.out_id:
                tmp_imgs = np.vstack([data,
                                      get_color_pallete(data, label,
                                                        self.dataset.get_class_colors(), self.config.data.background),
                                      get_color_pallete(data, pred,
                                                        self.dataset.get_class_colors(), self.config.data.background)])
                out_images = tmp_imgs if out_images is None else np.hstack(out_images, tmp_imgs)
            else:
                out_images = None

            tmp_res = edict
            tmp_res.hist = tmp_hist
            tmp_res.loss = loss
            tmp_res.out_img = out_images
            self.results_queue.put_nowait(tmp_res)

    def summarizing(self, all_results):
        hist = np.zeros((self.ndata, self.ndata))
        loss = 0.0
        out_imgs = []
        for res in all_results:
            hist += res.hist
            loss += res.loss
            if res.out_img is not None:
                out_imgs.append(res.out_img)
        acc = np.diag(hist).sum() / hist.sum()
        # acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        # freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        result = edict()
        result.acc = acc
        result.mean_iu = mean_iu

        return loss / len(all_results), result, out_imgs

    def run(self, model):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):
            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info(
                'GPU %s handle %d data.' % (device, len(shred_list)))
            p = self.context.Process(target=self.worker,
                                     args=(model, shred_list, device))
            procs.append(p)

        for p in procs:
            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)

        for p in procs:
            p.join()

        loss, result, out_imgs = self.summarizing(all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))

        return loss, result, out_imgs

    # evaluate the whole image at once
    def whole_eval(self, img, label, output_size, input_size=None, device=None):
        if input_size is not None:
            img, label, margin = self.process_image(img, label, input_size)
        else:
            img, label = self.process_image(img, label, input_size)

        pred, loss = self.val_func_process(img, label, device)

        if input_size is not None:
            pred = pred[:, margin[0]:(pred.shape[1] - margin[1]),
                   margin[2]:(pred.shape[2] - margin[3])]

        pred = pred.permute(1, 2, 0)
        pred = pred.cpu().numpy()

        if output_size is not None:
            pred = cv2.resize(pred,
                              (output_size[1], output_size[0]),
                              interpolation=cv2.INTER_LINEAR)

        pred = pred.argmax(2)

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

        scale_pred /= len(self.multi_scales)
        pred = scale_pred.argmax(2)
        loss = scale_loss / len(self.multi_scales)

        return pred, loss

    def scale_process(self, img, label, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, label_data, margin = self.process_image(img, crop_size)
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
                    label_sub = label_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, label_data, tmargin = self.process_image(img_sub, label_sub, crop_size)
                    temp_score, tmp_loss = self.val_func_process(input_data, label_data, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
                    data_loss += tmp_loss

            score = data_scale / count_scale
            data_loss = data_loss / count_loss
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output, data_loss

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
