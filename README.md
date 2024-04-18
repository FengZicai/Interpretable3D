# Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds

>[Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds]([https://arxiv.org/abs/2307.14605](https://ojs.aaai.org/index.php/AAAI/article/view/27944)) <br>
>[Tuo Feng](https://orcid.org/0000-0001-5882-3315), [Ruijie Quan](https://scholar.google.com/citations?user=WKLRPsAAAAAJ&hl=en), [Xiaohan Wang](https://scholar.google.com/citations?hl=zh-CN&user=iGA10XoAAAAJ), [Wenguan Wang](https://sites.google.com/view/wenguanwang),  [Yi Yang](https://scholar.google.com/citations?hl=zh-CN&user=RMSuNFwAAAAJ&view_op=list_works)

This is the official implementation of "Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds" (Accepted at AAAI 2024).


## Abstract
3D decision-critical tasks urgently require research on explanations to ensure system reliability and transparency. Extensive explanatory research has been conducted on 2D images, but there is a lack in the 3D field. Furthermore, the existing explanations for 3D models are post-hoc and can be misleading, as they separate explanations from the original model. To address these issues, we propose an ad-hoc interpretable classifier for 3D point clouds (i.e., Interpretable3D). As an intuitive case-based classifier, Interpretable3D can provide reliable ad-hoc explanations without any embarrassing nuances. It allows users to understand how queries are embedded within past observations in prototype sets. Interpretable3D has two iterative training steps: 1) updating one prototype with the mean of the embeddings within the same sub-class in Prototype Estimation, and 2) penalizing or rewarding the estimated prototypes in Prototype Optimization. The mean of embeddings has a clear statistical meaning, i.e., class sub-centers. Moreover, we update prototypes with their most similar observations in the last few epochs. Finally, Interpretable3D classifies new samples according to prototypes. We evaluate the performance of Interpretable3D on four popular point cloud models: DGCNN, PointNet2, PointMLP, and PointNeXt. Our Interpretable3D demonstrates comparable or superior performance compared to softmax-based black-box models in the tasks of 3D shape classification and part segmentation. 

## Dataset

ModelNet40: **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

ScanObjectNN: **ScanObjectNN** [here](https://hkust-vgd.github.io/scanobjectnn/) and save in `data/ScanObjectNN/main_split`.

Note: We conduct experiments on the hardest variant of ScanObjectNN (PB_T50_RS).


## Run

To train Interpretable3D-M:

```shell
python train_XXXX_ip3d.py
```

To test Interpretable3D-M:

```shell
python test_XXXX_ip3d.py
```

## Models

### Interpretable3D-M
Interpretable3D-M thoroughly updates the prototype with the mean of subclass centers obtained by online clustering.


### Pointnet++

Please refer to [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) for its installation.

#### Performance on ModelNet40:

| Model | Accuracy |
|--|--|
| PointNet2_MSG (Pytorch with normal) | [92.8](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/log/classification/pointnet2_msg_normals/logs/pointnet2_cls_msg.txt)|
| Interpretable3D-M+PointNet2_MSG (Pytorch with normal) | [93.5](PointNet2/log/classification/pointnet2_cls_msg_ip3d/logs/pointnet2_cls_msg_ip3d.txt) |
| Interpretable3D-M+PointNet2_MSG (Pytorch with normal) (vote) |  **[93.7](PointNet2/log/classification/pointnet2_cls_msg_ip3d/eval.txt)**|


#### Performance on ScanObjectNN

| Model | OA(%) | mAcc(%) |
|--|--|--|
| PointNet2_MSG (Pytorch with normal) (reprod.) | [79.9](PointNet2\log\classification\pointnet2_cls_msg_scanobjectnn\logs\pointnet2_cls_msg.txt) | [77.1](PointNet2\log\classification\pointnet2_cls_msg_scanobjectnn\logs\pointnet2_cls_msg.txt) |
| Interpretable3D-M+PointNet2_MSG (Pytorch with normal) (reprod.) | **[80.0](PointNet2\log\classification\pointnet2_cls_msg_scanobjectnn_ip3d\logs\pointnet2_cls_msg.txt)** | **[77.3](PointNet2\log\classification\pointnet2_cls_msg_scanobjectnn_ip3d\logs\pointnet2_cls_msg.txt)**|


More code and experimental results will be gradually open-sourced.


## Discussion
We find that training the softmax classifier and Interpretable3D-M simultaneously leads to faster convergence and better results.

## Citation

If you find the code useful in your research, please consider citing our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/27944):

```BibTeX
@inproceedings{feng2024interpretable3D,
	title={Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds},
	author={Feng, Tuo and Quan, Ruijie and Wang, Xiaohan and Wang, Wenguan, and Yang, Yi},
	booktitle=AAAI,
	year={2024}
}
```

Any comments, please email: feng.tuo@student.uts.edu.au.


## Acknowledgments
We thank for the opensource codebases: [DNC](https://github.com/ChengHan111/DNC), [ProtoSeg](https://github.com/tfzhou/ProtoSeg), [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), and [Cluster3DSeg](https://github.com/FengZicai/Cluster3DSeg). 
