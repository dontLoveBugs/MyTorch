# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/6 20:33
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : ade20k.py
"""

import os
import os.path as osp
import numpy as np
import scipy.io as sio
import time
import cv2

import torch

from modules.datasets.seg.BaseDataset import BaseDataset


class ADE(BaseDataset):

    def _fetch_data(self, img_path, gt_path, dtype=np.float32):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        gt = gt - 1  # 0:ingore --> -1:ignore

        return img, gt

    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        img_name = item
        gt_name = item.split('.')[0] + ".png"

        return img_name, gt_name

    def get_class_colors(self, *args):
        color_list = sio.loadmat(osp.join(self.root, 'color150.mat'))
        color_list = color_list['colors']
        color_list = color_list[:, ::-1, ]
        color_list = np.array(color_list).astype(int).tolist()
        return color_list

    def _get_pairs(self):
        img_paths = []
        mask_paths = []
        if self.split == 'train':
            img_folder = osp.join(self.root, 'images/training')
            mask_folder = osp.join(self.root, 'annotations/training')
        else:
            img_folder = osp.join(self.root, 'images/validation')
            mask_folder = osp.join(self.root, 'annotations/validation')

        file_list = os.listdir(img_folder)
        file_list.sort()  # sort to order
        for filename in file_list:
            basename, _ = osp.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = osp.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = osp.join(mask_folder, maskname)
                if osp.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)

        return img_paths, mask_paths

    @classmethod
    def get_class_names(*args):
        return ['wall', 'building, edifice', 'sky',
                'floor, flooring', 'tree', 'ceiling', 'road, route',
                'bed ', 'windowpane, window ',
                'grass', 'cabinet', 'sidewalk, pavement',
                'person, individual, someone, somebody, mortal, soul',
                'earth, ground', 'door, double door', 'table',
                'mountain, mount', 'plant, flora, plant life',
                'curtain, drape, drapery, mantle, pall', 'chair',
                'car, auto, automobile, machine, motorcar', 'water',
                'painting, picture', 'sofa, couch, lounge', 'shelf',
                'house',
                'sea', 'mirror', 'rug, carpet, carpeting', 'field',
                'armchair', 'seat', 'fence, fencing', 'desk',
                'rock, stone',
                'wardrobe, closet, press', 'lamp',
                'bathtub, bathing tub, bath, tub', 'railing, rail',
                'cushion', 'base, pedestal, stand', 'box',
                'column, pillar',
                'signboard, sign',
                'chest of drawers, chest, bureau, dresser',
                'counter', 'sand', 'sink', 'skyscraper',
                'fireplace, hearth, open fireplace',
                'refrigerator, icebox', 'grandstand, covered stand',
                'path',
                'stairs, steps', 'runway',
                'case, display case, showcase, vitrine',
                'pool table, billiard table, snooker table',
                'pillow',
                'screen door, screen', 'stairway, staircase',
                'river',
                'bridge, span', 'bookcase', 'blind, screen',
                'coffee table, cocktail table',
                'toilet, can, commode, crapper, pot, potty, stool, throne',
                'flower', 'book', 'hill', 'bench', 'countertop',
                'stove, kitchen stove, range, kitchen range, cooking stove',
                'palm, palm tree', 'kitchen island',
                'computer, computing machine, computing device, data processor, electronic computer, information processing system',
                'swivel chair', 'boat', 'bar', 'arcade machine',
                'hovel, hut, hutch, shack, shanty',
                'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle',
                'towel', 'light, light source', 'truck, motortruck',
                'tower',
                'chandelier, pendant, pendent',
                'awning, sunshade, sunblind',
                'streetlight, street lamp',
                'booth, cubicle, stall, kiosk',
                'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box',
                'airplane, aeroplane, plane', 'dirt track',
                'apparel, wearing apparel, dress, clothes', 'pole',
                'land, ground, soil',
                'bannister, banister, balustrade, balusters, handrail',
                'escalator, moving staircase, moving stairway',
                'ottoman, pouf, pouffe, puff, hassock',
                'bottle', 'buffet, counter, sideboard',
                'poster, posting, placard, notice, bill, card',
                'stage', 'van', 'ship', 'fountain',
                'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
                'canopy',
                'washer, automatic washer, washing machine',
                'plaything, toy',
                'swimming pool, swimming bath, natatorium',
                'stool', 'barrel, cask', 'basket, handbasket',
                'waterfall, falls', 'tent, collapsible shelter',
                'bag',
                'minibike, motorbike', 'cradle', 'oven', 'ball',
                'food, solid food', 'step, stair',
                'tank, storage tank',
                'trade name, brand name, brand, marque',
                'microwave, microwave oven', 'pot, flowerpot',
                'animal, animate being, beast, brute, creature, fauna',
                'bicycle, bike, wheel, cycle ', 'lake',
                'dishwasher, dish washer, dishwashing machine',
                'screen, silver screen, projection screen',
                'blanket, cover', 'sculpture', 'hood, exhaust hood',
                'sconce',
                'vase', 'traffic light, traffic signal, stoplight',
                'tray',
                'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
                'fan', 'pier, wharf, wharfage, dock', 'crt screen',
                'plate', 'monitor, monitoring device',
                'bulletin board, notice board', 'shower',
                'radiator',
                'glass, drinking glass', 'clock', 'flag']


class ValADE(ADE):

    def __init__(self, root, local_rank, world_size,
                 split='test', mode=None, preprocess=None):
        super(ValADE, self).__init__(root, split=split,
                                     mode=mode, preprocess=preprocess)
        self.local_rank = local_rank
        self.world_size = world_size

        self.set_device_pairs()

    def set_device_pairs(self):
        stride = int(np.ceil(self.get_length() / self.world_size))

        e_record = min((self.local_rank + 1) * stride, self.get_length())
        # print('start:', self.local_rank * stride, ' end:', e_record)
        self.images_path = self.images_path[self.local_rank * stride: e_record]
        self.gts_path = self.gts_path[self.local_rank * stride: e_record]


if __name__ == '__main__':
    # ade_train = ADE(root='/data/wangxin/ADEChallengeData2016', split='train')
    # ade_val = ADE(root='/data/wangxin/ADEChallengeData2016', split='val')

    # from torch.utils.data.dataloader import DataLoader
    # d_train = DataLoader(ade_train, batch_size=1)
    # d_val = DataLoader(ade_val, batch_size=1, shuffle=False)
    #
    # print('num of train:', len(d_train))
    # print('num of test:', len(d_val))
    #
    # for i, d in enumerate(d_train):
    #     print(i, d['fn'], d['data'].shape, d['label'].shape)
    #
    #
    # for i, d in enumerate(d_val):
    #     print(i, d['fn'], d['data'].shape, d['label'].shape)

    # for i in range(ade_train.get_length()):
    #     print(i, ade_train[i]['fn'])

    # print(ade_train.get_class_colors())

    # print(ade_train.get_length())
    # print(ade_val.get_length())

    color_list = sio.loadmat('color150.mat')
    color_list = color_list['colors']
    print(color_list)
    print('---------')
    color_list = color_list[:, ::-1, ]
    print(color_list)
    color_list = np.array(color_list).astype(int).tolist()
    color_list.insert(0, [0, 0, 0])
    print('---------')
    print(color_list)
    print(len(color_list))
