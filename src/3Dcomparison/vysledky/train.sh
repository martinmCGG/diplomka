#!/bin/bash

while getopts ":m:w:" opt; do
  case ${opt} in
    m )
      model=$OPTARG
      ;;
    w)
      weights=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))


case $model in
MVCNN)
	 echo "Training MVCNN"
	 cd ~/models/MVCNN
	 python train.py --caffemodel=alexnet_imagenet.npy --weights=tmp/model.ckpt-$weights
	 ;;
PNET) 
	echo "Training PNET"
	cd ~/models/PNET
	python train.py --weights log/model.ckpt-$weights
	 ;;
PNET2) 
	echo "Training PNET2"
	cd ~/models/PNET2
	python train_multi_gpu.py --weights log/model.ckpt-$weights
	 ;;
SEQ2SEQ) 
	cd ~/models/SEQ2SEQ
	echo "Training SEQ2SEQ" 
	python run.py --train True --weights logs/mvmodel.ckpt-$weights
	;;
*) 
	echo "Error: Unknown Model"
	 ;;
esac




#python evaluate.py --model_path=log/model250.ckpt

cd ~/models/SEQ2SEQ
#python train.py --weights=logs/mvmodel.ckpt-30 --train=False

cd ~/models/vysledky
#python3 MakeConfusionMatrix.py
#python3 MakeTable.py

cd ~/models
