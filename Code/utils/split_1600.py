# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
import shutil
import random

unlabeled_imgs_dir = '../../Dataset/TrainingSet/LungInfection-Train/Pseudo-label/Imgs'
save_path = '../../Dataset/TrainingSet/LungInfection-Train/Pseudo-label/DataPrepare/Imgs_splits'

img_list = os.listdir(unlabeled_imgs_dir)
random.shuffle(img_list)

print(img_list)

for i in range(0, int(len(img_list)/5)):
    choosed_img_lits = img_list[i*5:i*5+5]
    for img_name in choosed_img_lits:
        save_split_path = os.path.join(save_path, 'image_{}'.format(i))
        os.makedirs(save_split_path, exist_ok=True)
        shutil.copyfile(os.path.join(unlabeled_imgs_dir, img_name),
                        os.path.join(save_split_path, img_name))
    print('[Processed] image_{}: {}'.format(i, choosed_img_lits))