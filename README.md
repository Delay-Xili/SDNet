# SDNet — Revisiting Sparse Convolutional Model for Visual Recognition

This repository contains the official PyTorch implementation of the paper: 
[Xili Dai*](https://delay-xili.github.io/), Mingyang Li*, [Shengbang Tong*](https://tsb0601.github.io/petertongsb/),
[Pengyuan Zhai](https://billyzz.github.io/), Xingjian Gao,
[Shao-Lun Huang](https://sites.google.com/view/slhuang/), [Zhihui Zhu](https://www.cis.jhu.edu/~zhihui/index.html), 
[Chong You](https://sites.google.com/view/cyou), [Yi Ma](https://people.eecs.berkeley.edu/~yima/). 
["Revisiting Sparse Convolutional Model for Visual Recognition"](https://arxiv.org/abs/2210.12945). NeurIPS 2022.

## Introduction
Despite strong empirical performance for image classification, 
deep neural networks are often regarded as ``black boxes'' and they are difficult to interpret. 
On the other hand, sparse convolutional models, which assume that a signal can be expressed by a linear combination of a few elements from a convolutional dictionary, 
are powerful tools for analyzing natural images with good theoretical interpretability and biological plausibility. 
However, such principled models have not demonstrated competitive performance when compared with empirically designed deep networks. 
This paper revisits the sparse convolutional modeling for image classification and bridges the gap between good empirical performance (of deep learning) and good interpretability (of sparse convolutional models). 
Our method uses differentiable optimization layers that are defined from convolutional sparse coding as drop-in replacements of standard convolutional layers in conventional deep neural networks. 
We show that such models have equally strong empirical performance on CIFAR-10, CIFAR-100 and ImageNet datasets when compared to conventional neural networks. 
By leveraging stable recovery property of sparse modeling, we further show that such models can be much more robust to input corruptions as well as adversarial perturbations in testing through a simple proper trade-off between sparse regularization and data reconstruction terms.



## Reproducing Results

### Installation for Reproducibility

For ease of reproducibility, we suggest you install `Miniconda` (or `Anaconda` if you prefer) before executing the following commands.

```bash
git clone https://github.com/Delay-Xili/SDNet
cd SDNet
conda create -y -n sdnet
source activate sdnet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
mkdir data logs
```


## Here, we will release code and checkpoints in the near future! Stay tuned!
