#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg -n 1 >& logs/vgg/out1.txt &
CUDA_VISIBLE_DEVICES=1 th train.lua --model vgg_bn_drop -s logs/vgg -n 2 >& logs/vgg/out2.txt &
CUDA_VISIBLE_DEVICES=2 th train.lua --model vgg_bn_drop -s logs/vgg -n 4 >& logs/vgg/out4.txt &
CUDA_VISIBLE_DEVICES=3 th train.lua --model vgg_bn_drop -s logs/vgg -n 8 >& logs/vgg/out8.txt &


