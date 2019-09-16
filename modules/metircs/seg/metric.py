# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/8 16:49
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : metric.py
"""

# Original Author: Donny You(youansheng@gmail.com)
# Segmentation pspnet_ade score.


import numpy as np
from easydict import EasyDict as edict


class SegMetric(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        # 0<= class_value < n_class
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)

        return hist

    def update(self, label_preds, label_trues):
        """
        :param label_preds: numpy.array  HXW
        :param label_trues: numpy.array  HXW
        :return:
        """
        assert label_preds.shape == label_trues.shape
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # cls_iu = dict(zip(range(self.n_classes), iu))

        # return acc, acc_cls, fwavacc, mean_iu, cls_iu
        # result = edict()
        # result.acc = acc
        # result.acc_cls = acc_cls
        # result.fwavacc = fwavacc
        # result.mean_iu = mean_iu
        # result.cls_iu = cls_iu
        return [acc, mean_iu]

    def get_mean_iou(self):
        return self._get_scores()['mean_iu']

    def get_pixel_acc(self):
        return self._get_scores()['acc']

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


# operations
np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))

    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                       minlength=n_cl ** 2).reshape(n_cl,
                                                    n_cl), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    mean_pixel_acc = correct / labeled
    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc
