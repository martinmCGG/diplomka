#Training script
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model to train')
parser.add_argument('--weights',default='', help='Path to pretrained model weights')
args = parser.parse_args()

model = args.model
command = 'cd ~/models/{}; '.format(model)

if model == 'MVCNN':
    command += 'python train.py --caffemodel=alexnet_imagenet.npy --weights=tmp/model.ckpt-{}'.format(args.weights)
elif model == 'PNET':
    command += 'python train.py --weights log/model.ckpt-{}'.format(args.weights)
elif model == 'PNET2':
    command += 'python train_multi_gpu.py --weights log/model.ckpt-{}'.format(args.weights)
elif model == 'SEQ2SEQ':
    command += 'python run.py --train True --weights logs/mvmodel.ckpt-{}'.format(args.weights)
    
print('Training {}'.format(model))

os.system(command)



