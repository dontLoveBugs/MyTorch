import cv2
import torch
import numpy as np
from torch.utils import data

from easydict import EasyDict as edict
from modules.utils.img_utils import random_scale, random_mirror, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape

"""
    重新实现数据读取
"""


class TrainPre(object):
    def __init__(self, config):
        assert isinstance(config, edict), 'config is not edict.'
        self.config = config

    def __call__(self, img, gt):
        img, gt = random_mirror(img, gt)
        if self.config.get('train_scale_array') is not None:
            img, gt, scale = random_scale(img, gt, self.config.train_scale_array)

        img = normalize(img, self.config.img_mean, self.config.img_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)

        p_img = p_img.transpose(2, 0, 1)
        p_gt = p_gt - 1

        extra_dict = None

        return p_img, p_gt, extra_dict


def get_train_loader(engine, dataset):
    train_preprocess = TrainPre(engine.config.data)
    train_dataset = dataset(engine.config.data.dataset_path, "train", train_preprocess)

    train_sampler = None
    is_shuffle = True
    batch_size = engine.config.train.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=engine.config.train.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler, train_dataset.get_length() / batch_size
