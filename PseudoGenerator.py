# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

# ---- base lib -----
import os
import argparse
from datetime import datetime
import cv2
import numpy as np
import random
import shutil
from scipy import misc
# ---- torch lib ----
import torch
from torch.autograd import Variable
import torch.nn.functional as F
# ---- custom lib ----
# NOTES: Here we nly provide Res2Net, you can also replace it with other backbones
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import get_loader, test_dataset
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
from Code.utils.format_conversion import binary2edge


def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def trainer(train_loader, model, optimizer, epoch, opt, total_step):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]    # replace your desired scale
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(images)
            # ---- loss function ----
            loss5 = joint_loss(lateral_map_5, gts)
            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = torch.nn.BCEWithLogitsLoss()(lateral_edge, edges)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 5 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, '
                  'lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    # ---- save model_lung_infection ----
    save_path = 'Snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'Semi-Inf-Net-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'Semi-Inf-Net-%d.pth' % (epoch+1))


def train_module(_train_path, _train_save, _resume_snapshot):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='epoch number')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default=_train_path)
    parser.add_argument('--train_save', type=str, default=_train_save)
    parser.add_argument('--resume_snapshot', type=str, default=_resume_snapshot)
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)
    model = Network(channel=32, n_class=1).cuda()

    model.load_state_dict(torch.load(opt.resume_snapshot))

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        trainer(train_loader=train_loader, model=model, optimizer=optimizer,
                epoch=epoch, opt=opt, total_step=total_step)


def inference_module(_data_path, _save_path, _pth_path):
    model = Network(channel=32, n_class=1)
    model.load_state_dict(torch.load(_pth_path))
    model.cuda()
    model.eval()

    os.makedirs(_save_path, exist_ok=True)
    # FIXME
    image_root = '{}/'.format(_data_path)
    # gt_root = '{}/mask/'.format(data_path)
    test_loader = test_dataset(image_root, image_root, 352)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

        res = lateral_map_2  # final segmentation
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(_save_path + '/' + name, res)


def movefiles(_src_dir, _dst_dir):
    os.makedirs(_dst_dir, exist_ok=True)
    for file_name in os.listdir(_src_dir):
        shutil.copyfile(os.path.join(_src_dir, file_name),
                        os.path.join(_dst_dir, file_name))


if __name__ == '__main__':
    slices = './Dataset/TrainingSet/LungInfection-Train/Pseudo-label/DataPrepare'
    slices_dir = slices + '/Imgs_split'
    slices_pred_seg_dir = slices + '/pred_seg_split'
    slices_pred_edge_dir = slices + '/pred_edge_split'

    # NOTES: Hybrid-label = Doctor-label + Pseudo-label
    semi = './Dataset/TrainingSet/LungInfection-Train/Pseudo-label/DataPrepare/Hybrid-label'
    semi_img = semi + '/Imgs'
    semi_mask = semi + '/GT'
    semi_edge = semi + '/Edge'

    if (not os.path.exists(semi_img)) or (len(os.listdir(semi_img)) != 50):
        shutil.copytree('Dataset/TrainingSet/LungInfection-Train/Doctor-label/Imgs',
                        semi_img)
        shutil.copytree('Dataset/TrainingSet/LungInfection-Train/Doctor-label/GT',
                        semi_mask)
        shutil.copytree('Dataset/TrainingSet/LungInfection-Train/Doctor-label/Edge',
                        semi_edge)
        print('Copy done')
    else:
        print('Check done')

    slices_lst = os.listdir(slices_dir)
    random.shuffle(slices_lst)

    print("#" * 20, "\nStart Training (Inf-Net)\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
                    "Infection Segmentation from CT Scans', 2020, arXiv.\n"
                    "----\nPlease cite the paper if you use this code and dataset. "
                    "And any questions feel free to contact me "
                    "via E-mail (gepengai.ji@163.com)\n----\n", "#" * 20)

    for i, split_name in enumerate(slices_lst):
        print('\n[INFO] {} ({}/320)'.format(split_name, i))
        # ---- inference ----
        test_aux_dir = os.path.join(slices_dir, split_name)
        test_aux_save_dir = os.path.join(slices_pred_seg_dir, split_name)
        if i == 0:
            snapshot_dir = './Snapshots/save_weights/Inf-Net/Inf-Net-100.pth'
        else:
            snapshot_dir = './Snapshots/semi_training/Semi-Inf-Net_{}/Semi-Inf-Net-10.pth'.format(i-1)

        inference_module(_data_path=test_aux_dir, _save_path=test_aux_save_dir, _pth_path=snapshot_dir)

        os.makedirs(os.path.join(slices_pred_edge_dir, split_name), exist_ok=True)
        for pred_name in os.listdir(test_aux_save_dir):
            edge_tmp = binary2edge(os.path.join(test_aux_save_dir, pred_name))
            cv2.imwrite(os.path.join(slices_pred_edge_dir, split_name, pred_name), edge_tmp)

        # ---- move generation ----
        movefiles(test_aux_dir, semi_img)
        movefiles(test_aux_save_dir, semi_mask)
        movefiles(os.path.join(slices_pred_edge_dir, split_name), semi_edge)

        # ---- training ----
        train_module(_train_path=semi,
                     _train_save='semi_training/Semi-Inf-Net_{}'.format(i),
                     _resume_snapshot=snapshot_dir)

    # move img/pseudo-label into `./Dataset/TrainingSet/LungInfection-Train/Pseudo-label`
    shutil.copytree(semi_img, './Dataset/TrainingSet/LungInfection-Train/Pseudo-label/Imgs')
    shutil.copytree(semi_mask, './Dataset/TrainingSet/LungInfection-Train/Pseudo-label/GT')
    shutil.copytree(semi_edge, 'Dataset/TrainingSet/LungInfection-Train/Pseudo-label/Edge')
    print('Pseudo Label Generated!')