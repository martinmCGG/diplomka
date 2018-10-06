#!/bin/bash

cd ~/models/MVCNN
source virtual/bin/activate
python test.py --weights=`pwd`/tmp/model.ckpt-200
deactivate


cd ~/models/vysledky
python3 MakeConfusionMatrix.py
python3 MakeTable.py