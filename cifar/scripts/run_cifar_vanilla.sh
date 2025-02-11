#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=48gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=24:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate torch

# sample scripts for training vanilla teacher models

python train_teacher.py --model wrn_40_2

python train_teacher.py --model resnet56

python train_teacher.py --model resnet110

python train_teacher.py --model resnet32x4

python train_teacher.py --model vgg13

python train_teacher.py --model ResNet50
