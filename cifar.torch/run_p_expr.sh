#!/usr/bin/env bash
export CUDNN_PATH="/home/shai/caffe_area/cudnn-7.5-linux-x64-v5.1/cuda/lib64/libcudnn.so.5"

run_expr () {
  let nodes=$1
  let lrd=$2
  let epoch=$3
  let merge_freq=$4
  let start_device=$5
  echo "running expr $nodes $lrd $epoch $merge_freq $start_device"
  
  for i in `seq 0 $(($nodes - 1))` 
  do
    let device=$(($start_device + $i))
    save="logs/vgg/n_$nodes/lrd_$lrd/sesop_freq_$merge_freq/id_$i"
    mkdir -p $save
    CUDA_VISIBLE_DEVICES=$(($device + $i)) th train_parallel.lua --model vgg_bn_drop \
      -s $save -backend cudnn -f $merge_freq --max_epoch $epoch --epoch_step $lrd -n $nodes -i $i >& $save/out.txt &
  done
} 

run_expr 1 25 300 666 0

exit
#1 node (sgd)
run_expr 1 4 100 500 0
run_expr 1 7 100 500 1
run_expr 1 13 100 500 2
run_expr 1 25 100 500 3
wait


#2 nodes
run_expr 2 4 100 500 0
wait
run_expr 2 7 100 500 2
wait
run_expr 2 13 100 500 0
wait
run_expr 2 25 100 500 2
wait


#4 nodes
run_expr 4 4 100 500 0
wait
run_expr 4 7 100 500 0
wait
run_expr 4 13 100 500 0
wait
run_expr 4 25 100 500 0
wait


###################################



#1 node (sgd)
run_expr 1 4 100 1000 0
run_expr 1 7 100 1000 1
run_expr 1 13 100 1000 2
run_expr 1 25 100 1000 3
wait


#2 nodes
run_expr 2 4 100 1000 0
run_expr 2 7 100 1000 2
wait
run_expr 2 13 100 1000 0
run_expr 2 25 100 1000 2
wait


#4 nodes
run_expr 4 4 100 1000 0
wait
run_expr 4 7 100 1000 0
wait
run_expr 4 13 100 1000 0
wait
run_expr 4 25 100 1000 0
wait

