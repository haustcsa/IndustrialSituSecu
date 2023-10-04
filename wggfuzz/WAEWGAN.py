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

from PreProcessing import dataPre
from baseDataset import baseDataset
from seq2seq import TextDataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")#训练次数
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")#批次大小一次训练所抓取的数据样本数量
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")#优化器adam的梯度的一阶动量衰减 momentum
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")#优化器adam的梯度的二阶动量衰减 momentum
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent code")# latent(潜)空间的维数, 可以理解为噪声数据的维度
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--data_len", type=int, default=33, help="interval between image sampling")

opt = parser.parse_args()
print(opt)
num = 0
img_shape = (opt.channels,opt.data_len, 257)

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

def gen_img_plot(model,epoch,test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.im

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):

        x = self.model(img)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 1024),
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



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# Use binary cross-entropy loss   使用BCEloss计算交叉熵损失
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modul = Encoder().to(device)
decoder = Decoder()

modul = Decoder().to(device)

discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()



data = dataPre()[0].float().to(device)

datasets = TextDataset(data)

dataloader = torch.utils.data.DataLoader(
    datasets,
    batch_size=opt.batch_size,
    shuffle=True,

)
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # print("zzz",z.shape)
    gen_imgs = decoder(z)
    gen_imgs = gen_imgs.view(opt.latent_dim,opt.data_len,257)
    softmax = nn.Softmax(dim=2)
    gen_imgs = softmax(gen_imgs)
    gen_imgs_onehot = torch.zeros_like(gen_imgs)
    gen_imgs_onehot.scatter_(2,gen_imgs.argmax(dim=2,keepdim=True),1)

    print(gen_imgs_onehot.shape)

    gen_imgs_onehot = torch.argmax((gen_imgs_onehot),dim=2)
    print("gen_imgs")
    print(gen_imgs_onehot)


    # int_tensor = torch.floor(gen_imgs).to(torch.int64)
    # gen_imgs = torch.where(int_tensor < 0, torch.tensor(0).to('cuda'), int_tensor)
    # one_hot_matrix = torch.nn.functional.one_hot(gen_imgs)
    # print(len(one_hot_matrix))
    # print(one_hot_matrix)

    # save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------




for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = decoder(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs.detach())

        # Gradient penalty
        alpha = Tensor(np.random.random((real_imgs.shape[0], 1, 1)))

        fake_imgs = fake_imgs.view(real_imgs.shape[0],  real_imgs.shape[1], real_imgs.shape[2])

        interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)

        d_interpolates = discriminator(interpolates)

        fake = Variable(Tensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        print(gradients.size())
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Wasserstein loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Train the generator every n_critic iterations
        if i % 5 == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = decoder(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(gen_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )



        if epoch%10==9 and i ==624:
            print("11111111111111")
            num = 1
            torch.save(decoder.state_dict(), 'modelwae66'+str(num)+'.pth')
