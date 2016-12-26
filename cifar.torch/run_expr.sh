#!/usr/bin/env bash

export CUDNN_PATH="/home/shai/caffe_area/cudnn-7.5-linux-x64-v5.1/cuda/lib64/libcudnn.so.5"

CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 1 >& logs/vgg/out1.txt &
CUDA_VISIBLE_DEVICES=1 th train.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 2 >& logs/vgg/out2.txt &
#CUDA_VISIBLE_DEVICES=2 th train.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 4 >& logs/vgg/out4.txt &
#CUDA_VISIBLE_DEVICES=3 th train.lua --model vgg_bn_drop -s logs/vgg -backend cudnn -n 8 >& logs/vgg/out8.txt &


