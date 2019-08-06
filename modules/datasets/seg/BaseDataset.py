# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/6 11:38
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


import os
import time
import cv2
import torch
import numpy as np

import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, root, split='test', mode=None, preprocess=None):
        super(BaseDataset, self).__init__()
        self.split = split
        self.root = root
        self.mode = mode if mode is not None else split
        self.images_path, self.gts_path = self._get_pairs()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path, gt_path = self.images_path[index], self.gts_path[index]
        item_name = img_path.split("/")[-1].split(".")[0]
        img, gt = self._fetch_data(self.images_path[index], self.gts_path[index])

        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        # print('img:', img.shape, ' gt:', gt.shape)

        if self.split is 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name),
                           n=len(self.images_path))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)

        return img, gt

    def _get_pairs(self):
        raise NotImplementedError

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)

        return img

    @classmethod
    def get_class_colors(*args):
        raise NotImplementedError

    @classmethod
    def get_class_names(*args):
        raise NotImplementedError


if __name__ == "__main__":
    data_setting = {'img_root': '',
                    'gt_root': '',
                    'train_source': '',
                    'eval_source': ''}
    bd = BaseDataset(data_setting, 'train', None)
    print(bd.get_class_names())
