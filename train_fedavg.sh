#!/bin/bash
#PBS -l nodes=2:xeon:ppn=2
#PBS -l walltime=24:00:00
#PBS -e error.txt
#PBS -o output.txt

cd /home/u93525/Documents/federated-learning-gan/PyTorch-GAN/implementations/dcgan

/glob/development-tools/versions/oneapi/2022.1.1/oneapi/mpi/2021.5.0/bin/mpiexec -n 5 /home/u93525/.conda/envs/fedml/bin/python dcgan.py --n_epochs=10 --sample_interval=100
