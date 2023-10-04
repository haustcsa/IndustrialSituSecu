import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib as plt
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch

latent_dim=100
img_shape = (1,33, 257)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh(),
        )

    def forward(self, z):
        # print("ssdad",z.shape)
        img_flat = self.model(z)

        # img = img_flat.view(73, 257)

        return img_flat


if __name__== "__main__" :
    decoder = Decoder()
    decoder.load_state_dict(torch.load('modelwae6610.pth'), strict=False)
    for i in range(100):

        # decoder.eval()
        z = Variable(Tensor(np.random.normal(0, 1, (10 ** 2, latent_dim))))
        gen_imgs = decoder(z)
        gen_imgs = gen_imgs.view(100,33,257)
        softmax = nn.Softmax(dim=2)
        gen_imgs = softmax(gen_imgs)
        gen_imgs_onehot = torch.zeros_like(gen_imgs)
        gen_imgs_onehot.scatter_(2,gen_imgs.argmax(dim=2,keepdim=True),1)

        print(gen_imgs_onehot.shape)

        gen_imgs_onehot = torch.argmax((gen_imgs_onehot),dim=2)
        print("gen_imgs")

        gen_imgs_onehot = gen_imgs_onehot.numpy()
        print(gen_imgs_onehot)
        vfunc = np.vectorize(hex)
        gen_imgs_onehot = vfunc(gen_imgs_onehot)
        print(gen_imgs_onehot)
        gen_imgs_onehot = gen_imgs_onehot.tolist()
        gen_imgs_onehot = [[x for x in subseq if x != '0x100'] for subseq in gen_imgs_onehot]
        gen_imgs_onehot = [[x[:2]+'0'+x[2:] if len(x)==3 else x for x in subseq ] for subseq in gen_imgs_onehot]
        print(gen_imgs_onehot)
        hex_str = [''.join([s[2:] for s in hex_list]) for hex_list in gen_imgs_onehot]

        print(hex_str)

        # 打开一个文本文件，并将每个字符串写入文件
        with open('F:\\outputwae6610.txt', 'a') as f:
            for s in hex_str:
                f.write(s + '\n')


