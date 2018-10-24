#Testing script
import argparse
import os
import re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='ALL', help='model to train')
parser.add_argument('--weights',default='', help='Path to pretrained model weights')

args = parser.parse_args()

models = ['MVCNN', 'PNET', 'PNET2', 'SEQ2SEQ','KDNET']

model_commands={
    'MVCNN': 'python test.py --weights=logs/model.ckpt-',
    'PNET' : 'python evaluate.py --model_path=logs/model.ckpt-',
    'PNET2': 'python evaluate.py --model_path=logs/model.ckpt-',
    'SEQ2SEQ':'python train.py --train=False --weights=logs/mvmodel.ckpt-',
    'KDNET':'python train_batch.py --test=True --weights=logs/model.pth-'
    }

model_weights = {}
if args.weights == '':
    for model in models:
        files = os.listdir('/home/krabec/models/{}/logs'.format(model))
        if model != 'KDNET':
            files = ['.'.join(file.split('.')[0:-1]) for file in files if re.match('.*index$', file)]
        newest = max([int(x.split('-')[-1]) for x in files])
        model_weights[model] = newest
    
else:
    for model in models:
        model_weights[model] = args.weights

for model in models:
    if args.model == model or args.model == 'ALL':
        command = 'cd ~/models/{}; '.format(model)
        command += model_commands[model]+str(model_weights[model])
        print(command)
        os.system(command)


#Test via docker
models = ['VRNENS']
model_commands={
    'VRNENS': 'sh docker_script.sh',
    }
for model in models:
    if args.model == model or args.model == 'ALL':
        command = 'cd ~/dockers/{}; '.format(model)
        command += model_commands[model]
        print(command)
        os.system(command)

os.system('cd ~/models/vysledky; python3 MakeConfusionMatrix.py')
