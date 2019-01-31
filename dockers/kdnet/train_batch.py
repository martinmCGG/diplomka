from __future__ import print_function
from datasets import PartDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
from kdtree import make_cKDTree
from kdnet import KDNet_Batch
import sys
import os
import modelnet_dataset
import modelnet_h5_dataset   
from Logger import Logger

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights',default=None, type=int, help='Number of model to finetune or evaluate')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--num_points', type=int, default=2048, help='Number of points')
parser.add_argument('--data', type=str, default='/data/converted', help='Path to data')
parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
parser.add_argument('--max_epoch', type=int, default=50, help='Number of points')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
args =  parser.parse_args()


def split_ps(point_set):
    #print point_set.size()
    num_points = point_set.size()[0]//2
    diff = point_set.max(dim=0)[0] - point_set.min(dim=0)[0]
    dim = torch.max(diff, dim = 0)[1].item()#[0,0]
    cut = torch.median(point_set[:,dim])[0][0]
    
    left = torch.nonzero(point_set[:,dim] > cut)
    left_idx = torch.squeeze(left, len(left.size())-1)
    right = torch.nonzero(point_set[:,dim] < cut)
    right_idx = torch.squeeze(right ,len(right.size())-1)
    middle_idx = torch.squeeze(torch.nonzero(point_set[:,dim] == cut),0)

    if torch.numel(left_idx) < num_points+2:
        size = num_points - torch.numel(left_idx)
        to_cat = torch.zeros((),dtype=torch.long).new_full((size,),middle_idx[0][0])
        left_idx = torch.cat([left_idx, to_cat],0)
    if torch.numel(right_idx) < num_points:
        size = num_points - torch.numel(right_idx)
        to_cat = torch.zeros((),dtype=torch.long).new_full((size,),middle_idx[0][0])
        right_idx = torch.cat([right_idx, to_cat],0)

    left_ps = torch.index_select(point_set, dim = 0, index = left_idx)
    right_ps = torch.index_select(point_set, dim = 0, index = right_idx)
    return left_ps, right_ps, dim
def split_ps_reuse(point_set, level, pos, tree, cutdim):
    sz = point_set.size()
    num_points = np.array(sz)[0]/2
    max_value = point_set.max(dim=0)[0]
    min_value = -(-point_set).max(dim=0)[0]

    diff = max_value - min_value
    dim = torch.max(diff, dim = 0)[1].item()#[0,0]

    cut = torch.median(point_set[:,dim])[0][0]
    left = torch.nonzero(point_set[:,dim] > cut)
    left_idx = torch.squeeze(left, len(left.size())-1)
    right = torch.nonzero(point_set[:,dim] < cut)
    right_idx = torch.squeeze(right ,len(right.size())-1)
    middle_idx = torch.squeeze(torch.nonzero(point_set[:,dim] == cut),0)

    if torch.numel(left_idx) < num_points+2:
        size = num_points - torch.numel(left_idx)
        to_cat = torch.zeros((),dtype=torch.long).new_full((size,),middle_idx[0][0])
        left_idx = torch.cat([left_idx, to_cat],0)
    if torch.numel(right_idx) < num_points:
        size = num_points - torch.numel(right_idx)
        to_cat = torch.zeros((),dtype=torch.long).new_full((size,),middle_idx[0][0])
        right_idx = torch.cat([right_idx, to_cat],0)

    left_ps = torch.index_select(point_set, dim = 0, index = left_idx)
    right_ps = torch.index_select(point_set, dim = 0, index = right_idx)

    tree[level+1][pos * 2] = left_ps
    tree[level+1][pos * 2 + 1] = right_ps
    cutdim[level][pos * 2] = dim
    cutdim[level][pos * 2 + 1] = dim

    return


def train(args):
    start_epoch = 0
    net = KDNet_Batch().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)    
    batch_size = args.batch_size
    LOG_DIR = args.log_dir    
    
    if args.weights!=None:
        weights = int(args.weights)
        start_epoch = weights
        net.load_state_dict(torch.load(os.path.join(args.log_dir,'model-{}.pth'.format(weights))))
        ACC_LOGGER.load((os.path.join(args.log_dir,"kdnet_acc_train_accuracy.csv"),os.path.join(args.log_dir,"kdnet_acc_eval_accuracy.csv")), epoch=weights)
        LOSS_LOGGER.load((os.path.join(args.log_dir,"kdnet_loss_train_loss.csv"), os.path.join(args.log_dir,'kdnet_loss_eval_loss.csv')), epoch=weights)
    
    BASE_DIR = args.data  
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'train_files.txt'), batch_size=1, npoints=args.num_points, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'test_files.txt'), batch_size=1, npoints=args.num_points, shuffle=False)
    
    for epoch in range(start_epoch, start_epoch+args.max_epoch):
        eval(TEST_DATASET, net, args, epoch=epoch)
        train_one_epoch(TRAIN_DATASET, net, optimizer, epoch, args)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(args.log_dir,'model-{}.pth'.format(epoch)))
        ACC_LOGGER.save(LOG_DIR)
        LOSS_LOGGER.save(LOG_DIR)
        ACC_LOGGER.plot(dest=LOG_DIR)
        LOSS_LOGGER.plot(dest=LOG_DIR)
  
def forward(DATASET, net, args, test=False):
    levels = (np.log(args.num_points)/np.log(2)).astype(int)
    points_batch = []
    cutdim_batch = []
    targets = []
    bt = args.batch_size
    start = time.time()
    exit = False
    for batch in range(bt):
        #j = np.random.randint(l)
        #point_set, class_label = d[j]
        if not DATASET.has_next_batch():
            exit = True    
            DATASET.reset()
            DATASET.has_next_batch()
            break;
        batch_data, class_label = DATASET.next_batch(augment= not test)
        point_set = np.reshape(batch_data,(args.num_points,3))
        cutdim, tree = make_cKDTree(point_set, depth=levels)
        targets.append(torch.tensor(class_label).long())
        cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]
        points = torch.stack((torch.FloatTensor(tree[-1]),))
        points_batch.append(torch.unsqueeze(torch.squeeze(points), 0).transpose(2,1))
        cutdim_batch.append(cutdim_v)
    
    points_v = Variable(torch.cat(points_batch, 0)).cuda()
    target_v = Variable(torch.cat(targets, 0)).cuda()
    cutdim_processed = []
    for i in range(len(cutdim_batch[0])):
        cutdim_processed.append(torch.stack([item[i] for item in cutdim_batch], 0))
    pred = net(points_v, cutdim_processed[::-1])
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target_v.data).cpu().sum()
    loss = F.nll_loss(pred, target_v)
    return pred_choice.tolist(), target_v.tolist(), loss, correct, exit
      
def backward(loss, optimizer):
    loss.backward()
    optimizer.step()

def train_one_epoch(dataset, net, optimizer, epoch, args):
    it = 0
    losses = []
    corrects = [] 
    batch_size = args.batch_size
    while True:
        it+=1
        optimizer.zero_grad()
        _, _, loss, correct, exit = forward(dataset, net, args)
        losses.append(loss.item())
        corrects.append(correct.item())
        backward(loss, optimizer)
        if it%10 == 0:
            loss = np.mean(losses)
            acc =  sum(corrects) /  float(10*batch_size)
            print('epoch: %d, loss: %f, acc %f' %(epoch, loss , acc))       
            LOSS_LOGGER.log(loss, epoch, "train_loss")
            ACC_LOGGER.log(acc, epoch, "train_accuracy")
            losses = []
            corrects = []  
        if exit:
            break

def eval(dataset, net, args, epoch = None):
    size = dataset.size
    losses = []
    corrects = [] 
    predictions = []
    labels = []
    batch_size = args.batch_size
    while True:
        preds, labs, loss, correct, exit = forward(dataset, net, args, test=True)
        losses.append(loss.item())
        corrects.append(correct.item())
        predictions += preds
        labels += labs
        if exit:
            break    
    
    loss = np.mean(losses)
    acc =  sum(corrects) /  dataset.size
    if not args.test:
        print('EVAL: epoch: %d, loss: %f, acc %f' %(epoch, loss , acc))  
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
        
    else:
        print('EVAL: loss: %f, acc %f' %(loss , acc)) 
        import Evaluation_tools as et
        eval_file = os.path.join(args.log_dir, 'kdnet.txt')
        et.write_eval_file(args.data, eval_file, predictions , labels , 'KDNET')
        et.make_matrix(args.data, eval_file, args.log_dir)
    
  
if args.test:
    net = KDNet_Batch().cuda()
    net.load_state_dict(torch.load(os.path.join(args.log_dir,'model-{}.pth'.format(args.weights))))
    dataset = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(args.data, 'test_files.txt'), batch_size=1, npoints=args.num_points, shuffle=False)
    eval(dataset, net, args)
    #net.eval()     
else:
    LOSS_LOGGER = Logger("kdnet_loss")
    ACC_LOGGER = Logger("kdnet_acc")
    train(args)

