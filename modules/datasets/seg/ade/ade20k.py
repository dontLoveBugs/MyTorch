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
        color_list.insert(0, [0, 0, 0])
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
        for filename in os.listdir(img_folder):
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

        img_paths.sort(), mask_paths.sort()
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


if __name__ == '__main__':
    ade_train = ADE(root='/data/wangxin/ADEChallengeData2016', split='train')
    ade_val = ADE(root='/data/wangxin/ADEChallengeData2016', split='val')

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

    for i in range(ade_train.get_length()):
        print(i, ade_train[i]['fn'])

    print(ade_train.get_class_colors())


