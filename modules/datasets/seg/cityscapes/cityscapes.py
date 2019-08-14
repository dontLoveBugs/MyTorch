# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/14 15:53
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
 @File    : cityscapes.py
"""


import os
from modules.datasets.seg import BaseDataset


class CityScapes(BaseDataset):

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
        return [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]

    @classmethod
    def get_class_names(*args):
        # class counting(gtFine)
        # 2953 2811 2934  970 1296 2949 1658 2808 2891 1654 2686 2343 1023 2832
        # 359  274  142  513 1646
        return ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle']


if __name__ == '__main__':
    city_train = CityScapes(root='/data/wangxin/Dataset/CityScapes', split='train')
    city_val = CityScapes(root='/data/wangxin/Dataset/CityScapes', split='val')

    print(city_train.get_length())

    for i in range(city_train.get_length()):
        print(city_train[i]['fn'])