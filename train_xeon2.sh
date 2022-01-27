#!/bin/bash
#PBS -l nodes=2:xeon:ppn=2
#PBS -l walltime=24:00:00
#PBS -e error2.txt
#PBS -o output2.txt

cd /home/u93525/Documents/federated-learning-gan/pytorch-gan-federated-gan-2/implementations/dcgan

# /glob/development-tools/versions/oneapi/2022.1.1/oneapi/mpi/2021.5.0/bin/mpiexec -n 5 /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/pytorch-1.8.0/bin/python dcgan.py --n_epochs=1 --batch_size1=64 --batch_size2=64 --latent_dim=100 --img_size=32 

/glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/pytorch-1.8.0/bin/python dcgan.py --log_dir=logs2/ --models_dir=saved_models2/ --images_dir=images_fedavg2/  --n_epochs=10 --batch_size1=64 --batch_size2=64 --sample_interval=200 --noise_multiplier=1e-3

