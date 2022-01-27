#!/bin/bash
#PBS -l nodes=2:arria10:ppn=2
#PBS -l walltime=24:00:00
#PBS -e error.txt
#PBS -o output.txt

cd /home/u93525/Documents/federated-learning-gan/PyTorch-GAN/implementations/dcgan

/glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/pytorch-1.8.0/bin/python dcgan.py --n_epochs=10 --batch_size=32 --latent_dim=100 --img_size=32 

