#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from modules.engine.seg.config import Config
from run.network import PSPNet

from modules.utils.pyt_utils import parse_devices
from modules.utils.visualize import print_iou, show_img
from modules.engine.seg import Evaluator
from modules.engine.logger import get_logger
from modules.ops.seg.metric import hist_info, compute_score
from modules.datasets.seg.ade import ADE


logger = get_logger()
# read trainig confi file
config = Config(config_file='./config.json', train=False).get_config()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        label = label - 1

        pred = self.sliding_eval(img, config.eval.crop_size,
                                 config.eval.stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.data.num_classes, pred, label)
        results_dict = {'hist': hist_tmp,
                        'labeled': labeled_tmp,
                        'correct': correct_tmp}

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.data.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.data.num_classes, config.data.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(), True)
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    mp_ctx = mp.get_context('spawn')
    network = PSPNet(config.data.num_classes, criterion=None)
    dataset = ADE(config.data.dataset_path, 'val', None)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.data.num_classes,
                                 config.data.image_mean,
                                 config.data.image_std, network,
                                 config.eval.scale_array, config.eval.flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.log.snapshot_dir, args.epochs, config.log.val_log_file,
                      config.log.link_val_log_file)
