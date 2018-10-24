#!/bin/bash

while getopts ":m:w:" opt; do
  case ${opt} in
    m )
      m=$OPTARG
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

if [ "$m" == "MVCNN" ] || [ "$m" == "ALL" ]; then
  cd ~/models/MVCNN
  python test.py --weights=`pwd`/tmp/model.ckpt-43000
fi

if [ "$m" == "PNET" ] || [ "$m" == "ALL" ]; then
	cd ~/models/PNET
	python evaluate.py --model_path=log/model.ckpt-0
fi

if [ "$m" == "PNET2" ] || [ "$m" == "ALL" ]; then
	cd ~/models/PNET2
	python evaluate.py --model_path=log/model.ckpt-0
fi

if [ "$m" == "SEQ2SEQ" ] || [ "$m" == "ALL" ]; then
	cd ~/models/SEQ2SEQ
	python train.py --weights=logs/mvmodel.ckpt-50 --train=False
fi

cd ~/models/vysledky
python3 MakeConfusionMatrix.py
#python3 MakeTable.py

