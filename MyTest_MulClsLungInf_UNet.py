# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
import numpy as np
from Code.utils.dataloader_MulClsLungInf_UNet import LungDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from Code.model_lung_infection.InfNet_UNet import *  # use U-Net for multi-class segmentation
from scipy import misc
from Code.utils.split_class import split_class
import shutil


def inference(num_classes, input_channels, snapshot_dir, save_path):
    test_dataset = LungDataset(
        imgs_path='./Dataset/TestingSet/MultiClassInfection-Test/Imgs/',
        pseudo_path='./Results/Lung infection segmentation/Semi-Inf-Net/',  # NOTES: generated from `Semi-Inf-Net`
        label_path='./Dataset/TestingSet/MultiClassInfection-Test/GT/',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])]),
        is_test=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lung_model = Inf_Net_UNet(input_channels, num_classes).cuda()
    print(lung_model)
    lung_model.load_state_dict(torch.load(snapshot_dir))
    lung_model.eval()

    for index, (img, pseudo, img_mask, name) in enumerate(test_dataloader):
        img = img.to(device)
        pseudo = pseudo.to(device)
        img_mask = img_mask.to(device)

        output = lung_model(torch.cat((img, pseudo), dim=1))
        output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
        b, _, w, h = output.size()
        _, _, w_gt, h_gt = img_mask.size()

        # output b*n_class*h*w -- > b*h*w
        pred = output.cpu().permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, w, h).numpy().squeeze()
        print('Class numbers of prediction in total:', np.unique(pred))
        # pred = misc.imresize(pred, size=(w_gt, h_gt))
        os.makedirs(save_path, exist_ok=True)
        misc.imsave(save_path + name[0].replace('.jpg', '.png'), pred)
        split_class(save_path, name[0].replace('.jpg', '.png'), w_gt, h_gt)

    shutil.rmtree(save_path)
    print('Test done!')


if __name__ == "__main__":
    inference(num_classes=3,
              input_channels=6,
              snapshot_dir='./Snapshots/save_weights/Semi-Inf-Net_UNet/unet_model_200.pkl',
              save_path='./Results/Multi-class lung infection segmentation/class_12/'
              )
