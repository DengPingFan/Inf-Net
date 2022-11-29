# Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images

> **Authors:** 
> [Deng-Ping Fan](https://dengpingfan.github.io/), 
> [Tao Zhou](https://taozh2017.github.io/), 
> [Ge-Peng Ji](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), 
> [Yi Zhou](https://scholar.google.com/citations?hl=zh-CN&user=EnDCJKMAAAAJ), 
> [Geng Chen](https://www.researchgate.net/profile/Geng_Chen13), 
> [Huazhu Fu](http://hzfu.github.io/), 
> [Jianbing Shen](http://iitlab.bit.edu.cn/mcislab/~shenjianbing), and 
> [Ling Shao](http://www.inceptioniai.org/).

## 0. Preface

- This repository provides code for "_**Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images**_" TMI-2020. 
([arXiv Pre-print](https://arxiv.org/abs/2004.14133) & [medrXiv](https://www.medrxiv.org/content/10.1101/2020.04.22.20074948v1) & [中译版](http://dpfan.net/wp-content/uploads/TMI20_InfNet_Chinese_Finalv2.pdf))

- If you have any questions about our paper, feel free to contact us. And if you are using COVID-SemiSeg Dataset, 
Inf-Net or evaluation toolbox for your research, please cite this paper ([BibTeX](#8-citation)).

- We elaborately collect COVID-19 imaging-based AI research papers and datasets [awesome-list](https://github.com/HzFu/COVID19_imaging_AI_paper_list).

### 0.1. :fire: NEWS :fire:
- [2022/04/08] :boom: We release a new large-scale dataset on **Video Polyp Segmentation (VPS)** task, please enjoy it. [ProjectLink](https://github.com/GewelsJI/VPS)/ [PDF](https://arxiv.org/abs/2203.14291).
- [2021/04/15] Update the results on multi-class segmentation task, including 'Semi-Inf-Net & FCN8s' and 'Semi-Inf-Net & MC'. (Download link: [Google Drive](https://drive.google.com/file/d/1mIA9ggiftwhdzSAMl2sIAl3rkupt2_AY/view?usp=sharing))
- [2020/10/25] Uploading [中文翻译版](https://dengpingfan.github.io/papers/[2020][TMI]InfNet_Chinese.pdf).
- [2020/10/14] Updating the legend (1 * 1 -> 3 * 3; 3 * 3 -> 1 * 1) of Fig.3 in our manuscript. 2020/08/15 Updating equation (2) in our manuscript. <br>
  R_i = C(f_i, Dow(e_att)) * A_i -> R_i = C(f_i * A_i, Dow(e_{att}));
- [2020/08/15] Optimizing the testing code, now you can test the custom data without `gt_path`
- [2020/05/15] Our paper is accepted for publication in IEEE TMI
- [2020/05/13] :boom: Upload pre-trained weights. (Uploaded by Ge-Peng Ji)
- [2020/05/12] :boom: Release training/testing/evaluation code. (Updated by Ge-Peng Ji)
- [2020/05/01] Create repository.

### 0.2. Table of Contents

- [Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images](#inf-net-automatic-covid-19-lung-infection-segmentation-from-ct-images)
  - [0. Preface](#0-preface)
    - [0.1. :fire: NEWS :fire:](#01-fire-news-fire)
    - [0.2. Table of Contents](#02-table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1. Task Descriptions](#11-task-descriptions)
  - [2. Proposed Methods](#2-proposed-methods)
    - [2.1. Inf-Net](#21-inf-net)
      - [2.1.1 Overview](#211-overview)
      - [2.1.2. Usage](#212-usage)
    - [2.2. Semi-Inf-Net](#22-semi-inf-net)
      - [2.2.1. Overview](#221-overview)
      - [2.2.2. Usage](#222-usage)
    - [2.3. Semi-Inf-Net + Multi-class UNet](#23-semi-inf-net--multi-class-unet)
      - [2.3.1. Overview](#231-overview)
      - [2.3.2. Usage](#232-usage)
  - [3. Evaluation Toolbox](#3-evaluation-toolbox)
    - [3.1. Introduction](#31-introduction)
    - [3.2. Usage](#32-usage)
  - [4. COVID-SemiSeg Dataset](#4-covid-semiseg-dataset)
    - [3.1. Training set](#31-training-set)
    - [3.2. Testing set](#32-testing-set)
  - [4. Results](#4-results)
    - [4.1. Download link:](#41-download-link)
  - [5. Visualization Results:](#5-visualization-results)
  - [6. Paper list of COVID-19 related (Update continue)](#6-paper-list-of-covid-19-related-update-continue)
  - [7. Manuscript](#7-manuscript)
  - [8. Citation](#8-citation)
  - [9. LICENSE](#9-license)
  - [10. Acknowledgements](#10-acknowledgements)
  - [11. TODO LIST](#11-todo-list)
  - [12. FAQ](#12-faq)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## 1. Introduction

### 1.1. Task Descriptions

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/COVID19-Infection-1.png"/> <br />
    <em> 
    Figure 1. Example of COVID-19 infected regions in CT axial slice, where the red and green masks denote the 
    ground-glass opacity (GGO) and consolidation, respectively. The images are collected from [1].
    </em>
</p>

> [1] COVID-19 CT segmentation dataset, link: https://medicalsegmentation.com/covid19/, accessed: 2020-04-11.

## 2. Proposed Methods

- **Preview:**

    Our proposed methods consist of three individual components under three different settings: 

    - Inf-Net (Supervised learning with segmentation).
    
    - Semi-Inf-Net (Semi-supervised learning with doctor label and pseudo label)
    
    - Semi-Inf-Net + Multi-Class UNet (Extended to Multi-class Segmentation, including Background, Ground-glass Opacities, and Consolidation).
   
- **Dataset Preparation:**

    Firstly, you should download the testing/training set ([Google Drive Link](https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM)) 
    and put it into `./Dataset/` repository.

- **Download the Pretrained Model:**

    ImageNet Pre-trained Models used in our paper (
    [VGGNet16](https://download.pytorch.org/models/vgg16-397923af.pth), 
    [ResNet](https://download.pytorch.org/models/resnet50-19c8e357.pth), and 
    [Res2Net](https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth)), 
    and put them into `./Snapshots/pre_trained/` repository.

- **Configuring your environment (Prerequisites):**

    Note that Inf-Net series is only tested on Ubuntu OS 16.04 with the following environments (CUDA-10.0). 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n SINet python=3.6`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.
    
    - Installing [THOP](https://github.com/Lyken17/pytorch-OpCounter) for counting the FLOPs/Params of model.

### 2.1. Inf-Net

#### 2.1.1 Overview

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/Inf-Net.png"/> <br />
    <em> 
    Figure 2. The architecture of our proposed Inf-Net model, which consists of three reverse attention 
    (RA) modules connected to the paralleled partial decoder (PPD).
    </em>
</p>

#### 2.1.2. Usage

1. Train

    - We provide multiple backbone versions (see this [line](https://github.com/DengPingFan/Inf-Net/blob/0df52ec8bb75ef468e9ceb7c43a41b0886eced3a/MyTrain_LungInf.py#L133)) in the training phase, i.e., ResNet, Res2Net, and VGGNet, but we only provide the Res2Net version in the Semi-Inf-Net. Also, you can try other backbones you prefer to, but the pseudo labels should be RE-GENERATED with corresponding backbone.
    
    - Turn off the semi-supervised mode (`--is_semi=False`) turn off the flag of whether use pseudo labels (`--is_pseudo=False`) in the parser of `MyTrain_LungInf.py` and just run it! (see this [line](https://github.com/DengPingFan/Inf-Net/blob/0df52ec8bb75ef468e9ceb7c43a41b0886eced3a/MyTrain_LungInf.py#L121))

1. Test
    
    - When training is completed, the weights will be saved in `./Snapshots/save_weights/Inf-Net/`. 
    You can also directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=19p_G8NS4NwF4pZOEOX3w06ZLVh0rj3VW).
    
    - Assign the path `--pth_path` of trained weights and `--save_path` of results save and in `MyTest_LungInf.py`.
    
    - Just run it and results will be saved in `./Results/Lung infection segmentation/Inf-Net`

### 2.2. Semi-Inf-Net

#### 2.2.1. Overview

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/Semi-InfNet.png"/> <br />
    <em> 
    Figure 3. Overview of the proposed Semi-supervised Inf-Net framework.
    </em>
</p>

#### 2.2.2. Usage

1. Data Preparation for a pseudo-label generation. (Optional)
    
    - Dividing the 1600 unlabeled image into 320 groups (1600/K groups, we set K=5 in our implementation), 
    in which images with `*.jpg` format can be found in `./Dataset/TrainingSet/LungInfection-Train/Pseudo-label/Imgs/`. 
    (I suppose you have downloaded all the train/test images following the instructions [above](#2-proposed-methods))
    Then you only just run the code stored in `./SrcCode/utils/split_1600.py` to split it into multiple sub-dataset, 
    which are used in the training process of pseudo-label generation. The 1600/K sub-datasets will be saved in 
    `./Dataset/TrainingSet/LungInfection-Train/Pseudo-label/DataPrepare/Imgs_split/`
       
    - **You can also skip this process and download them from [Google Drive](https://drive.google.com/open?id=1Rn_HhgTEhy-M7qmPVspdJiamWC_kueFJ) that is used in our implementation.**

1. Generating Pseudo Labels (Optional)

    - After preparing all the data, just run `PseudoGenerator.py`. It may take at least day and a half to finish the whole generation.
    
    - **You can also skip this process and download intermediate generated file from [Google Drive](https://drive.google.com/open?id=1Rn_HhgTEhy-M7qmPVspdJiamWC_kueFJ) that is used in our implementation.**
    
    - When training is completed, the images with pseudo labels will be saved in `./Dataset/TrainingSet/LungInfection-Train/Pseudo-label/`.

1. Train

    - Firstly, turn off the semi-supervised mode (`--is_semi=False`) and turn on the flag of whether using pseudo labels
     (`--is_pseudo=True`) in the parser of `MyTrain_LungInf.py` and modify the path of training data to the pseudo-label 
     repository (`--train_path='Dataset/TrainingSet/LungInfection-Train/Pseudo-label'`). Just run it!
    
    - When training is completed, the weights (trained on pseudo-label) will be saved in `./Snapshots/save_weights/Inf-Net_Pseduo/Inf-Net_pseudo_100.pth`. Also, you can directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=1NMM0BoVJU9DS8u4yTG0L-V_mcHKajrmN). Now we have prepared the weights that is pre-trained on 1600 images with pseudo labels. Please note that these valuable images/labels can promote the performance and the stability of the training process, because of ImageNet pre-trained models are just designed for general object classification/detection/segmentation tasks initially.
    
    - Secondly, turn on the semi-supervised mode (`--is_semi=True`) and turn off the flag of whether using pseudo labels
     (`--is_pseudo=False`) in the parser of `MyTrain_LungInf.py` and modify the path of training data to the doctor-label (50 images)
     repository (`--train_path='Dataset/TrainingSet/LungInfection-Train/Doctor-label'`). Just run it.

1. Test
    
    - When training is completed, the weights will be saved in `./Snapshots/save_weights/Semi-Inf-Net/`. 
    You also can directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=1iHMQ9bjm4-qZaZZFXigZ-iruZOSB9mty).
    
    - Assign the path `--pth_path` of trained weights and `--save_path` of results save and in `MyTest_LungInf.py`.
    
    - Just run it! And results will be saved in `./Results/Lung infection segmentation/Semi-Inf-Net`.

### 2.3. Semi-Inf-Net + Multi-class UNet

#### 2.3.1. Overview

Here, we provide a general and simple framework to address the multi-class segmentation problem. We modify the 
original design of UNet that is used for binary segmentation, and thus, we name it as _Multi-class UNet_. 
More details can be found in our paper.

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/MultiClassInfectionSeg.png"/> <br />
    <em> 
    Figure 3. Overview of the proposed Semi-supervised Inf-Net framework.
    </em>
</p>

#### 2.3.2. Usage

1. Train

    - Just run `MyTrain_MulClsLungInf_UNet.py`
    
    - Note that `./Dataset/TrainingSet/MultiClassInfection-Train/Prior` is just borrowed from `./Dataset/TestingSet/LungInfection-Test/GT/`, 
    and thus, two repositories are equally.

1. Test
    
    - When training is completed, the weights will be saved in `./Snapshots/save_weights/Semi-Inf-Net_UNet/`. Also, you can directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=1V4VJl5X4kJVD6HHWYsGzbrbPh8Q_Vc0I).
    
    - Assigning the path of weights in parameters `snapshot_dir` and run `MyTest_MulClsLungInf_UNet.py`. All the predictions will be saved in `./Results/Multi-class lung infection segmentation/Consolidation` and `./Results/Multi-class lung infection segmentation/Ground-glass opacities`.

## 3. Evaluation Toolbox

### 3.1. Introduction

We provide a one-key evaluation toolbox for LungInfection Segmentation tasks, including Lung-Infection and Multi-Class-Infection. 
Please download the evaluation toolbox [Google Drive](https://drive.google.com/open?id=1BGUUmrRPOWPxdxnawFnG9TVZd8rwLqCF).

### 3.2. Usage

- Prerequisites: MATLAB Software (Windows/Linux OS is both works, however, we suggest you test it in the Linux OS for convenience.)

- run `cd ./Evaluation/` and `matlab` open the Matlab software via terminal

- Just run `main.m` to get the overall evaluation results.

- Edit the parameters in the `main.m` to evaluate your custom methods. Please refer to the instructions in the `main.m`.

## 4. COVID-SemiSeg Dataset

We also build a semi-supervised COVID-19 infection segmentation (**COVID-SemiSeg**) dataset, with 100 labelled CT scans 
from the COVID-19 CT Segmentation dataset [1] and 1600 unlabeled images from the COVID-19 CT Collection dataset [2]. 
Our COVID-SemiSeg Dataset can be downloaded at [Google Drive](https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM).

> [1]“COVID-19 CT segmentation dataset,” https://medicalsegmentation.com/covid19/, accessed: 2020-04-11.
> [2]J. P. Cohen, P. Morrison, and L. Dao, “COVID-19 image data collection,” arXiv, 2020.


### 3.1. Training set

1. Lung infection which consists of 50 labels by doctors (Doctor-label) and 1600 pseudo labels generated (Pseudo-label) 
by our Semi-Inf-Net model. [Download Link](http://dpfan.net/wp-content/uploads/LungInfection-Train.zip).

1. Multi-Class lung infection which also composed of 50 multi-class labels (GT) by doctors and 50 lung infection 
labels (Prior) generated by our Semi-Inf-Net model. [Download Link](http://dpfan.net/wp-content/uploads/MultiClassInfection-Train.zip).


### 3.2. Testing set

1. The Lung infection segmentation set contains 48 images associated with 48 GT. [Download Link](http://dpfan.net/wp-content/uploads/LungInfection-Test.zip).

1. The Multi-Class lung infection segmentation set has 48 images and 48 GT. [Download Link](http://dpfan.net/wp-content/uploads/MultiClassInfection-Test.zip).

1. The download link ([Google Drive](https://drive.google.com/file/d/1WHHIJ1R6DRV9maUD6qKT_-aiDktKLd0q/view?usp=sharing)) of our 638-dataset, which is used in Table.V of our paper.

== Note that ==: In our manuscript, we said that the total testing images are 50. However, we found there are two images with very small resolution and black ground-truth. Thus, we discard these two images in our testing set. The above link only contains 48 testing images. 


## 4. Results

To compare the infection regions segmentation performance, we consider the two state-of-the-art models U-Net and U-Net++. 
We also show the multi-class infection labeling results in Fig. 5. As can be observed, 
our model, Semi-Inf-Net & FCN8s, consistently performs the best among all methods. It is worth noting that both GGO and 
consolidation infections are accurately segmented by Semi-Inf-Net & FCN8s, which further demonstrates the advantage of 
our model. In contrast, the baseline methods, DeepLabV3+ with different strides and FCNs, all obtain unsatisfactory 
results, where neither GGO nor consolidation infections can be accurately segmented.

### 4.1. Download link:

Lung infection segmentation results can be downloaded from this [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EU4MU-DHvzZFuX1SRw3vwOwBSQsf2K4VsNrL-Vurc0KM8Q?e=EWWXI7)

Multi-class lung infection segmentation can be downloaded from this [link](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EannSF0mFIxImTuzqbAZ3-gBusy6luZ23NFhTiT4Xsib6g?e=nfTc7x)

## 5. Visualization Results:

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/InfectionSeg.png"/> <br />
    <em> 
    Figure 4. Visual comparison of lung infection segmentation results.
    </em>
</p>

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/MultiClassInfectionSeg.png"/> <br />
    <em> 
    Figure 5. Visual comparison of multi-class lung infection segmentation results, where the red and green labels 
    indicate the GGO and consolidation, respectively.
    </em>
</p>

## 6. Paper list of COVID-19 related (Update continue)

> Ori GitHub Link: https://github.com/HzFu/COVID19_imaging_AI_paper_list

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/paper-list-cover.png"/> <br />
    <em> 
    Figure 6. This is a collection of COVID-19 imaging-based AI research papers and datasets.
    </em>
</p>

## 7. Manuscript
https://arxiv.org/pdf/2004.14133.pdf

## 8. Citation

Please cite our paper if you find the work useful: 

	@article{fan2020infnet,
	  author={Fan, Deng-Ping and Zhou, Tao and Ji, Ge-Peng and Zhou, Yi and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
	  journal={IEEE Transactions on Medical Imaging}, 
	  title={Inf-Net: Automatic COVID-19 Lung Infection Segmentation From CT Images}, 
	  year={2020},
	  volume={39},
	  number={8},
	  pages={2626-2637},
	  doi={10.1109/TMI.2020.2996645}
	}

 
## 9. LICENSE

- The **COVID-SemiSeg Dataset** is made available for non-commercial purposes only. Any comercial use should get formal permission first.

- You will not, directly or indirectly, reproduce, use, or convey the **COVID-SemiSeg Dataset** 
or any Content, or any work product or data derived therefrom, for commercial purposes.


## 10. Acknowledgements
 
We would like to thank the whole organizing committee for considering the publication of our paper in this special issue (Special Issue on Imaging-Based Diagnosis of COVID-19) of IEEE Transactions on Medical Imaging. For more papers refer to [Link](https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=9153182).

## 11. TODO LIST

> If you want to improve the usability of code or any other pieces of advice, please feel free to contact me directly ([E-mail](gepengai.ji@gmail.com)).

- [ ] Support `NVIDIA APEX` training.

- [ ] Support different backbones (
VGGNet (done), 
ResNet, 
[ResNeXt](https://github.com/facebookresearch/ResNeXt)
[Res2Net (done)](https://github.com/Res2Net/Res2Net-PretrainedModels), 
[iResNet](https://github.com/iduta/iresnet), 
and 
[ResNeSt](https://github.com/zhanghang1989/ResNeSt) 
etc.)

- [ ] Support distributed training.

- [ ] Support lightweight architecture and faster inference, like MobileNet, SqueezeNet.

- [ ] Support distributed training

- [ ] Add more comprehensive competitors.

## 12. FAQ

1. If the image cannot be loaded on the page (mostly in domestic network situations).

    [Solution Link](https://blog.csdn.net/weixin_42128813/article/details/102915578)

2. I tested the U-Net, however, the Dice score is different from the score in TABLE II (Page 8 of our manuscript). <br>
   Note that, our Dice score is the mean dice score rather than the max Dice score. You can use our evaluation toolbox [Google Drive](https://drive.google.com/open?id=1BGUUmrRPOWPxdxnawFnG9TVZd8rwLqCF). 
   The training set of each compared model (e.g., U-Net, Attention-UNet, Gated-UNet, Dense-UNet, U-Net++, Inf-Net (ours)) is 48 images rather than 48 images + 1600 images.

**[⬆ back to top](#0-preface)**
