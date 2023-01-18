# 简介

本代码以MMDectionV2为基础构建

## 安装

首先配置以下环境
```
python=3.8
torch==1.10.0+cu113
torchaudio==0.10.0
torchvision==0.11.1+cu113
```
安装CUDA和cuDNN，版本如下
```
CUDA=11.3
cuDNN=8.2
```
之后安装
```
mmcls==0.23.0
mmcv-full==1.4.5
```
pip安装本源码
```
cd det-master
pip install -v -e .
```

## 实验说明
训练
```
python tools/train.py configs/det-code/mobilev3-weightfpn.py
##config内为整体构建py文件，可自行更改权重生成位置、训练时间等
```
测试
```
python tools/test.py configs/det-code/sparse-rcnn.py 权重位置 --show
```

## 相关代码

configs/det-code/** 下为具体配置代码
- Atss Atss.py [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection][https://arxiv.org/abs/1912.02424]
- Dynamic rcnn dynamic_rcnn.py [Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training][https://arxiv.org/abs/2004.06002]
- sparse rcnn sparse_rcnn.py [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals][https://arxiv.org/abs/2011.12450] 
- yolov3 yolov3.py [YOLOv3: An Incremental Improvement][https://arxiv.org/abs/1804.02767]
- Faster rcnn 
   - fpn
     - weightfpn(steelfpn) 本文 mobilev3-weightfpn.py
