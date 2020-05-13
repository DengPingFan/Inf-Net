# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
import torch
from torch.utils.data import Dataset
import cv2
from Code.utils.onehot import onehot


class LungDataset(Dataset):
    def __init__(self, imgs_path, pseudo_path, label_path, transform=None, is_test=False):
        self.transform = transform
        self.imgs_path = imgs_path  # 'data/class3_images/'
        self.pseudo_path = pseudo_path
        self.label_path = label_path    # 'data/class3_label/'
        self.is_test = is_test

    def __len__(self):
        return len(os.listdir(self.imgs_path))

    def __getitem__(self, idx):
        # processing img
        img_name = os.listdir(self.imgs_path)[idx]
        # image path
        imgA = cv2.imread(self.imgs_path + img_name)
        imgA = cv2.resize(imgA, (352, 352))

        # processing pseudo
        imgC = cv2.imread(self.pseudo_path + img_name.split('.')[0] + '.png')
        imgC = cv2.resize(imgC, (352, 352))

        # processing label
        imgB = cv2.imread(self.label_path + img_name.split('.')[0] + '.png', 0)
        if not self.is_test:
            imgB = cv2.resize(imgB, (352, 352))
        img_label = imgB
        # print(np.unique(img_label))

        img_label[img_label == 38] = 1
        img_label[img_label == 75] = 2

        img_label_onehot = onehot(img_label, 3)  # w * H * n_class
        img_label_onehot = img_label_onehot.transpose(2, 0, 1)  # n_class * w * H

        onehot_label = torch.FloatTensor(img_label_onehot)
        if self.transform:
            imgA = self.transform(imgA)
            imgC = self.transform(imgC)

        return imgA, imgC, onehot_label, img_name


# if __name__ == '__main__':
#
#     for train_batch in train_dataloader:
#         print(train_batch)
#
#     for test_batch in test_dataloader:
#         print(test_batch)
