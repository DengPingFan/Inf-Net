# Inf-Net
code for "Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans" submit to TMI20.

## Task Description

## Abstract
Coronavirus Disease 2019 (COVID-19) spread globally in early 2020, causing the world to face an existential health crisis. Automated detection of lung infections from computed tomography (CT) images offers a great potential to augment the traditional healthcare strategy for tackling COVID-19. However, segmenting infected regions from CT scans faces several challenges, including high variation in infection characteristics, and low intensity contrast between infections and normal tissues. Further, collecting a large amount of data is impractical within a short time period, inhibiting the training of a deep model. To address these challenges, a novel COVID-19 Lung Infection Segmentation Deep Network (Inf-Net) is proposed to automatically identify infected regions from chest CT scans. In our Inf-Net, a parallel partial decoder is used to aggregate the high-level features and generate a global map. Then, the implicit reverse attention and explicit edgeattention are utilized to model the boundaries and enhance the representations. Moreover, to alleviate the shortage of labeled data, we present a semi-supervised segmentation framework based on a randomly selected propagation strategy, which only requires a few labeled images and leverages primarily unlabeled data. Our semi-supervised framework can improve the learning ability and achieve a higher performance. Extensive experiments on a COVID-19 infection dataset demonstrate that the proposed Inf-Net outperforms most cutting-edge segmentation models and advances the state-of-the-art performance.

## Training set

## Testing set

## Results

## Paper list (update continue)
https://github.com/HzFu/COVID19_imaging_AI_paper_list

## Manuscript

## Citation
Please cite our paper if you find the work useful: 

	@inproceedings{fan2020InfNet,
  	title={Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans},
  	author={Fan, Deng-Ping and Zhou, Tao and Ji, Ge-Peng and Zhou, Yi and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
  	booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  	year={2020}
	}
  
