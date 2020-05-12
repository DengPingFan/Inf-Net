import os
import shutil
# from libtiff import TIFF
from scipy import misc
import random
import cv2
import numpy as np


# def tif2png(_src_path, _dst_path):
#     """
#     Usage:
#         formatting `tif/tiff` files to `jpg/png` files
#     :param _src_path:
#     :param _dst_path:
#     :return:
#     """
#     tif = TIFF.open(_src_path, mode='r')
#     image = tif.read_image()
#     misc.imsave(_dst_path, image)


def data_split(src_list):
    """
    Usage:
        randomly spliting dataset
    :param src_list:
    :return:
    """
    counter_list = random.sample(range(0, len(src_list)), 550)

    return counter_list


def binary2edge(mask_path):
    """
    func1: threshold(src, thresh, maxval, type[, dst]) -> retval, dst
            https://www.cnblogs.com/FHC1994/p/9125570.html
    func2: Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges

    :param mask_path:
    :return:
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # if <0, pixel=0 else >0, pixel=255
    mask_edge = cv2.Canny(mask_binary, 10, 150)

    return mask_edge


def random_list(low, high, number):
    # [low, high)
    return np.random.choice(np.random.randint(low, high), number, replace=False)


def binaryMask(im_path):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)

    return mask_binary


if __name__ == '__main__':
    # NOTES: split dataset
    # random_l = random_list(0, 98, 50)  # train_list
    # print(len(random_l))
    # root_path_img = '../Dataset/COVID-19/new_image/'
    # root_path_gt = '../Dataset/COVID-19/new_label_123/'
    #
    # save_test_path_img = '../Dataset/Object-level-123/TestDataset/image/'
    # save_test_path_gt = '../Dataset/Object-level-123/TestDataset/mask/'
    #
    # save_train_path_img = '../Dataset/Object-level-123/TrainDataset/image/'
    # save_train_path_gt = '../Dataset/Object-level-123/TrainDataset/mask/'
    #
    # os.makedirs(save_test_path_img, exist_ok=True)
    # os.makedirs(save_test_path_gt, exist_ok=True)
    # os.makedirs(save_train_path_img, exist_ok=True)
    # os.makedirs(save_train_path_gt, exist_ok=True)
    #
    # for _idx, name in enumerate(os.listdir(root_path_img)):
    #     if _idx in random_l:
    #         # train
    #         img_save_path = save_train_path_img + name
    #         gt_save_path = save_train_path_gt + name.replace('.jpg', '.png')
    #     else:
    #         # test
    #         img_save_path = save_test_path_img + name
    #         gt_save_path = save_test_path_gt + name.replace('.jpg', '.png')
    #
    #     shutil.copyfile(os.path.join(root_path_img, name), img_save_path)
    #     shutil.copyfile(os.path.join(root_path_gt, name.replace('.jpg', '.png')), gt_save_path)

    src = '/media/nercms/NERCMS/GepengJi/COVID-19/Dataset/COVID-19/new_label_12/'
    save_test = '/media/nercms/NERCMS/GepengJi/COVID-19/Dataset/Object-level/TestDataset/COVID-19/mask_12/'
    save_train = '/media/nercms/NERCMS/GepengJi/COVID-19/Dataset/Object-level/TrainDataset/mask_12/'
    os.makedirs(save_test, exist_ok=True)
    os.makedirs(save_train, exist_ok=True)
    train_lst = os.listdir('/media/nercms/NERCMS/GepengJi/COVID-19/Dataset/Object-level/TrainDataset/mask')

    for name in os.listdir(src):
        if name in train_lst:
            binary_im = binaryMask(src+name)
            cv2.imwrite(save_train+name, binary_im)
            # shutil.copyfile(src+name, save_train+name)
        else:
            binary_im = binaryMask(src + name)
            cv2.imwrite(save_test+name, binary_im)
            # shutil.copyfile(src+name, save_test+name)



    # path = '/media/nercms/NERCMS/GepengJi/COVID-19/Dataset/Instance-level-12/Test/mask/'
    # save = '/media/nercms/NERCMS/GepengJi/COVID-19/Dataset/Instance-level-12/Test/edge/'
    # for img_name in os.listdir(path):
    #     img_path = path + img_name
    #     im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     mask_edge = cv2.Canny(im, 10, 150)
    #     cv2.imwrite(save+img_name, mask_edge)
    #     print(mask_edge.shape)