## Unbiased Manifold Augmentation for Coarse Class Subdivision

<center><img src="http://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/Fig1.jpg" width = "500"/></center>

By Baoming Yan, KE GAO, Bo Gao, Lin Wang, Jiang Yang, Xiaobo Li.

This repo is the official implementation of "Unbiased Manifold Augmentation for Coarse Class Subdivision"

## Updates

***06/07/2022***

Initial commits.

## Introduction

Class Subdivision (CCS) is important for many practical applications, where the training set originally annotated for a coarse class (e.g. bird) needs to further support its sub-classes recognition (e.g. swan, crow) with only very few fine-grained labeled samples. From the perspective of causal representation learning, these sub-classes inherit the same determinative factors of the coarse class, and their difference lies only in values. Therefore, to support the challenging CCS task with minimum fine-grained labeling cost, an ideal data augmentation method should generate abundant variants by manipulating these sub-class samples at the granularity of generating factors. For this goal, traditional data augmentation methods are far from sufficient. They often perform in highly-coupled image or feature space, thus can only simulate global geometric or photometric transformations. Leveraging the recent progress of factor-disentangled generators, Unbiased Manifold Augmentation (UMA) is proposed for CCS. With a controllable StyleGAN pre-trained for a coarse class, an approximate unbiased augmentation is conducted on the factor-disentangled manifolds for each sub-class, revealing the unbiased mutual information between the target sub-class and its determinative factors. Extensive experiments have shown that in the case of small data learning (less than 1\% fine-grained samples of commonly used), our UMA can achieve 10.37\% average improvement compared with existing data augmentation methods. On challenging tasks with severe bias, the accuracy is improved by up to 16.79%.

## Getting Started

### Progressive Sample Synthesis (PSS)

```
cd ./encoder4editing/scripts
bash inference.sh
```

### Progressive Robust Learning (PRL)

```
cd ./ccs_training/
bash train.sh
```

## Citing

```
todo
```
