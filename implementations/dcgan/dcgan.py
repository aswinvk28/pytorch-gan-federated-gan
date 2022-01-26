import argparse
from asyncio.log import logger
import os
import numpy as np
import math
import logging

# from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import ThreadPoolExecutor
# import mpi4py.MPI as MPI

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

torch.autograd.set_detect_anomaly(True)

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size1", type=int, default=64, help="size of the batches of fed 1")
parser.add_argument("--batch_size2", type=int, default=64, help="size of the batches of fed 2")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--predict", type=int, default=1, help="predicting the images")
parser.add_argument("--noise_type", type=str, default='gaussian', help="type of the noise")
parser.add_argument("--noise_multiplier", type=float, default=1e-4, help="multiplier of the noise")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

"""
Gaussian Noise added to Parameters
"""
def gaussian_noise(params):
    return torch.randn(params.size()) * opt.noise_multiplier

"""
Laplacian Noise added to Parameters
"""
def laplacian_noise(params, mean, scale):
    dist = torch.distributions.laplace.Laplace(mean, scale)
    return dist.sample(params.size())

"""
Filter Function to Generate Noise
"""
def generate_noise(params, noise_type):
    if noise_type == "gaussian":
        return gaussian_noise(params)
    elif noise_type == "laplacian":
        return laplacian_noise(params, 0, opt.noise_multiplier)

"""
Applying Type 1 Attack on the Gradients
"""
def fedavg(discriminator_param_list, generator_param_list, batch_sizes, 
           global_distriminator_params, global_generator_params, noise_type='gaussian'):
    total_samples_discriminator = 0
    for ii, d in enumerate(discriminator_param_list):
        sample_size = batch_sizes[ii]
        total_samples_discriminator += sample_size
    
    total_samples_generator = 0
    for ii, d in enumerate(generator_param_list):
        sample_size = batch_sizes[ii]
        total_samples_generator += sample_size
        
    with torch.no_grad():
        for ii, d in enumerate(discriminator_param_list): # each client
            for k in d.keys():
                p = d[k].float()
                p += generate_noise(p, noise_type)
                global_distriminator_params[k] = global_distriminator_params[k].float()
                if ii == 0:
                    global_distriminator_params[k] = p * batch_sizes[0] / total_samples_discriminator
                else:
                    global_distriminator_params[k] += p * batch_sizes[0] / total_samples_discriminator
    
    with torch.no_grad():
        for ii, d in enumerate(generator_param_list):
            for k in d.keys():
                p = d[k].float()
                p += generate_noise(p, noise_type)
                global_generator_params[k] = global_generator_params[k].float()
                if ii == 0:
                    global_generator_params[k] = p * batch_sizes[0] / total_samples_generator
                else:
                    global_generator_params[k] += p * batch_sizes[0] / total_samples_generator

    return global_distriminator_params, global_generator_params

# ----------
#  Training
# ----------

"""
Futures Function for MPI4PY
"""
def futures_call(imgs, optimizer_G, optimizer_D, opt, dataloader, logger, i, epoch):
    # Adversarial ground truths
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_imgs = Variable(imgs.type(Tensor))

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

    # Generate a batch of images
    gen_imgs = generator(z)
    
    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()
    
    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
    )
    
    loggers[logger].write(
        "\n[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
    )

    batches_done = epoch * len(dataloader) + i
    if batches_done % opt.sample_interval == 0:
        torch.save(generator.state_dict(), "saved_models/generator_%d-%d.pth" % (batches_done, epoch))
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d-%d.pth" % (batches_done, epoch))
        
    return discriminator.state_dict(), generator.state_dict()

def predict(named_parameters, batch_sizes, opt, dataloader, epoch, i):
    g = Generator()
    d = Discriminator()
    
    discriminator_state_dict, generator_state_dict = named_parameters
    
    # generator_state_dict = {}
    # for name, parameter in generator_state_dict.items():
    #     generator_state_dict[name] = parameter
        
    # discriminator_state_dict = {}
    # for name, parameter in discriminator_state_dict.items():
    #     discriminator_state_dict[name] = parameter
    
    g.load_state_dict(generator_state_dict)
    d.load_state_dict(discriminator_state_dict)
    
    z = Tensor(np.random.normal(0, 1, (batch_sizes[0], opt.latent_dim)))

    # Generate a batch of images
    gen_imgs = g(z)
    
    batches_done = epoch * len(dataloader) + i
    if batches_done % opt.sample_interval == 0:
        save_image(gen_imgs.data[:25], "images_fedavg/%d.png" % batches_done, nrow=5, normalize=True)

# loggers = []
# comm = MPI.COMM_WORLD
# for i in range(5):
#     mode = MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND
#     fh = MPI.File.Open(comm, "logfile"+str(i)+".log", mode)
#     fh.Set_atomicity(True)
#     loggers.append(fh)

loggers = []
for i in range(5):
    fh = open("logfile"+str(i)+".log", 'a')
    loggers.append(fh)

def log_output(i):
    loggers[i].write("\nNew information logged")

if __name__ == "__main__":

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader1 = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size1,
        shuffle=True,
    )

    dataloader2 = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size2,
        shuffle=True,
    )

    optimizers_G = []
    optimizers_D = []
    for i in range(5):
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizers_G.append(optimizer_G)
        optimizers_D.append(optimizer_D)

    buffer1 = [opt]*5
    buffer2 = [dataloader1, dataloader1, dataloader2, dataloader2, dataloader2]
    buffer3 = list(range(5))
    
    buffer4 = optimizers_G
    buffer5 = optimizers_D
    
    fedavg_params_discriminator, fedavg_params_generator = discriminator.state_dict(), generator.state_dict()
    
    data_loader2 = iter(dataloader2)
    for epoch in range(opt.n_epochs):
        for j, (imgs1, _) in enumerate(dataloader1):
            
            imgs2, _ = next(data_loader2)
            
            # with ThreadPoolExecutor(max_workers=5) as executor:
                # Optimizers
            discriminator_param_list = []
            generator_param_list = []
            
            buffer6 = [imgs1, imgs1, imgs2, imgs2, imgs2]
            
            for i in range(5):
                discriminator_params, generator_params = futures_call(
                    buffer6[i], buffer4[i], buffer5[i], buffer1[i], 
                    buffer2[i], buffer3[i], j, epoch
                )
                discriminator_param_list.append(discriminator_params)
                generator_param_list.append(generator_params)
                
            fedavg_params_discriminator, fedavg_params_generator = \
                fedavg(discriminator_param_list, generator_param_list, 
                [imgs1.shape[0], imgs1.shape[0]] + [imgs2.shape[0], imgs2.shape[0], imgs2.shape[0]], 
                fedavg_params_discriminator, fedavg_params_generator,
                    noise_type=opt.noise_type)

            predict((fedavg_params_discriminator, fedavg_params_generator), 
                    [imgs1.shape[0], imgs1.shape[0]] + [imgs2.shape[0], imgs2.shape[0], imgs2.shape[0]], 
                    opt, dataloader1, epoch, j)
                
