from __future__ import print_function
import os
import sys
import argparse
from prepare_data import create_data

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/data/converted', help='dataset used to train and test')
parser.add_argument('--test',action='store_true', help='train mode or test mode')
parser.add_argument('--weights', help='number of model to be finetuned or tested')
args = parser.parse_args()

NUM_CLASSES = 40
VIEWS = 12
BATCH_SIZE = 256
TRAIN_FOR_EPOCH = 100
SAVE_PERIOD = 5

train_cmd = 'python train.py --n_hidden=128 --decoder_embedding_size=256 --use_lstm=False --keep_prob=0.5   --learning_rate=0.0002  --n_max_keep_model=200 '
train_cmd += ' --training_epoches={} '.format(TRAIN_FOR_EPOCH)
train_cmd += ' --save_epoches={} '.format(SAVE_PERIOD)
train_cmd += ' --n_views={} '.format(VIEWS)
train_cmd += ' --batch_size={} '.format(BATCH_SIZE)

out = "./logs"
outfile = "results.csv"

path = ["train_features.npy", "train_labels.npy", "test_features.npy", "test_labels.npy"]
path = [os.path.join(args.data, p) for p in path]

if not os.path.isfile(path[0]):
    create_data(os.path.join(args.data,'train.txt'), 'train', args.data, VIEWS, batch_size=BATCH_SIZE)
    create_data(os.path.join(args.data,'test.txt'), 'test', args.data, VIEWS, batch_size=BATCH_SIZE)
     
               
train_cmd += '--n_classes={} '.format(NUM_CLASSES)

train_cmd += '--train_feature_file={} --train_label_file={} --test_feature_file={} --test_label_file={} '.format(path[0], path[1], path[2], path[3])

train_cmd += '--save_seq_embeddingmvmodel_path={} --checkpoint_path={} --test_acc_file={}'.format(os.path.join(out, "mvmodel.ckpt"), os.path.join(out, "checkpoint"),outfile )

if args.test: 
    train_cmd += " --test=True "
if args.weights:
    train_cmd +=" --weights={} ".format(args.weights)

print(train_cmd)
os.system(train_cmd)


