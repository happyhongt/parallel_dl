#!/usr/bin/env bash

export CUDNN_PATH="/home/shai/caffe_area/cudnn-7.5-linux-x64-v5.1/cuda/lib64/libcudnn.so.5"

mkdir -p logs/vgg/n_1/id_0
mkdir -p logs/vgg/n_2/id_0
mkdir -p logs/vgg/n_2/id_1

CUDA_VISIBLE_DEVICES=0 th train_parallel.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 2 -i 0 >& logs/vgg/n_2/id_0/out.txt &
CUDA_VISIBLE_DEVICES=1 th train_parallel.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 2 -i 1 >& logs/vgg/n_2/id_1/out.txt &




CUDA_VISIBLE_DEVICES=2 th train_parallel.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 1 -i 0 >& logs/vgg/n_1/id_0/out.txt &