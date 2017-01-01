#!/usr/bin/env bash

export CUDNN_PATH="/home/shai/caffe_area/cudnn-7.5-linux-x64-v5.1/cuda/lib64/libcudnn.so.5"

CUDA_VISIBLE_DEVICES=0 th train_parallel.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 2 -i 0 >& logs/vgg/out0.txt &
CUDA_VISIBLE_DEVICES=1 th train_parallel.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 2 -i 1 >& logs/vgg/out1.txt &




