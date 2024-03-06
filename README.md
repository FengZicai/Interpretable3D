# Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds
This is the official implementation of "Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds" (Accepted at AAAI 2024).


## Abstract
3D decision-critical tasks urgently require research on explanations to ensure system reliability and transparency. Extensive explanatory research has been conducted on 2D images, but there is a lack in the 3D field. Furthermore, the existing explanations for 3D models are post-hoc and can be misleading, as they separate explanations from the original model. To address these issues, we propose an ad-hoc interpretable classifier for 3D point clouds (i.e., Interpretable3D). As an intuitive case-based classifier, Interpretable3D can provide reliable ad-hoc explanations without any embarrassing nuances. It allows users to understand how queries are embedded within past observations in prototype sets. Interpretable3D has two iterative training steps: 1) updating one prototype with the mean of the embeddings within the same sub-class in Prototype Estimation, and 2) penalizing or rewarding the estimated prototypes in Prototype Optimization. The mean of embeddings has a clear statistical meaning, i.e., class sub-centers. Moreover, we update prototypes with their most similar observations in the last few epochs. Finally, Interpretable3D classifies new samples according to prototypes. We evaluate the performance of Interpretable3D on four popular point cloud models: DGCNN, PointNet2, PointMLP, and PointNeXt. Our Interpretable3D demonstrates comparable or superior performance compared to softmax-based black-box models in the tasks of 3D shape classification and part segmentation. 


```BibTeX
@inproceedings{feng2024interpretable3D,
	title={Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds},
	author={Feng, Tuo and Quan, Ruijie and Wang, Xiaohan and Wang, Wenguan, and Yang, Yi},
	booktitle=AAAI,
	year={2024}
}
```
