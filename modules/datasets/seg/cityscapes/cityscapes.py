# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/14 15:53
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : cityscapes.py
"""

import os
import cv2
import numpy as np

from modules.datasets.seg import BaseDataset


class CityScapes(BaseDataset):

    def __init__(self, root, split='test', mode=None, preprocess=None, ignore_label=255):
        super(CityScapes, self).__init__(root, split=split, mode=mode, preprocess=preprocess)

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def _fetch_data(self, img_path, gt_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

        tgt = self.id2trainId(gt).astype(np.uint8)

        return img, tgt

    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        img_name = item
        gt_name = item.split('.')[0] + ".png"

        return img_name, gt_name

    def _get_pairs(self):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            for root, _, files in os.walk(img_folder):
                files.sort()
                for filename in files:
                    if filename.endswith('.png'):
                        imgpath = os.path.join(root, filename)
                        foldername = os.path.basename(os.path.dirname(imgpath))
                        maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                        maskpath = os.path.join(mask_folder, foldername, maskname)
                        if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                            img_paths.append(imgpath)
                            mask_paths.append(maskpath)
                        else:
                            print('cannot find the mask or image:', imgpath, maskpath)
            print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
            return img_paths, mask_paths

        if self.split in ('train', 'val', 'test'):
            img_folder = os.path.join(self.root, 'leftImg8bit/' + self.split)
            mask_folder = os.path.join(self.root, 'gtFine/' + self.split)
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            return img_paths, mask_paths
        else:
            raise NotImplementedError

    @classmethod
    def get_class_colors(*args):
        # return [[128, 64, 128], [244, 35, 232], [70, 70, 70],
        #         [102, 102, 156], [190, 153, 153], [153, 153, 153],
        #         [250, 170, 30], [220, 220, 0], [107, 142, 35],
        #         [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
        #         [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
        #         [0, 0, 230], [119, 11, 32]]

        return [[128, 64, 128], [232, 35, 244], [70, 70, 70],
                [156, 102, 102], [153, 153, 190], [153, 153, 153],
                [30, 170, 250], [0, 220, 220], [35, 142, 107],
                [152, 251, 152], [180, 130, 70], [60, 20, 220], [0, 0, 250],
                [142, 0, 0], [70, 0, 0], [100, 60, 0], [100, 80, 0],
                [230, 0, 0], [32, 11, 119]]

    @classmethod
    def get_class_names(*args):
        # class counting(gtFine)
        # 2953 2811 2934  970 1296 2949 1658 2808 2891 1654 2686 2343 1023 2832
        # 359  274  142  513 1646
        return ['road', 'sidewalk', 'building',
                'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation',
                'terrain', 'sky', 'person', 'rider',
                'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle']


class ValCityScapes(CityScapes):

    def __init__(self, root, local_rank, world_size,
                 split='test', mode=None, preprocess=None, ignore_label=255):
        super(ValCityScapes, self).__init__(root, split=split,
                                            mode=mode, preprocess=preprocess,
                                            ignore_label=ignore_label)
        self.local_rank = local_rank
        self.world_size = world_size

        self.set_device_pairs()

    def set_device_pairs(self):
        stride = int(np.ceil(self.get_length() / self.world_size))

        e_record = min((self.local_rank + 1) * stride, self.get_length())
        self.images_path = self.images_path[self.local_rank * stride, e_record]
        self.gts_path = self.gts_path[self.local_rank * stride, e_record]


if __name__ == '__main__':
    city_train = CityScapes(root='/data/datasets/CityScapes', split='train')
    city_val = CityScapes(root='/data/datasets/CityScapes', split='val')

    print(city_train.get_length())

    for i in range(city_train.get_length()):
        print(city_train[i]['fn'])
        data = city_train[i]['data']
        label = city_train[i]['label']
        print(label[515][515])
        from modules.utils.visualize import get_color_pallete

        img = get_color_pallete(data, label, city_train.get_class_colors(), 255)

        import cv2

        cv2.imshow(city_train[i]['fn'], img)
        cv2.waitKey()
