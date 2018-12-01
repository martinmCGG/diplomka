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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights',default='-', help='Path to pretrained model weights')
parser.add_argument('--test', type=bool, default=False, help='Whether to test')
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

def count_and_exit(sum_correct, sum_sample, predictions, labels):
    '''sys.path.insert(0, '/home/krabec/models/vysledky')
    from MakeCategories import make_categories
    make_categories('/home/krabec/models/MVCNN/modelnet40v1', '/home/krabec/models/vysledky/kdnet.txt', predictions, labels, 'KDNET2')'''
    sys.exit()
    
    
num_points = 2048
test = args.test

levels = (np.log(num_points)/np.log(2)).astype(int)
net = KDNet_Batch().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

BASE_DIR = 'data'  
TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=1, npoints=num_points, shuffle=True)
TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=1, npoints=num_points, shuffle=False)

DATASET = TEST_DATASET if test else TRAIN_DATASET

batch_size = 128   
start_epoch = 0
if test:
    net.load_state_dict(torch.load(args.weights))
    net.eval()
elif args.weights[-1] != '-':
    net.load_state_dict(torch.load(args.weights))
    start_epoch = int(args.weights.split('-')[-1])

sum_correct = 0
sum_sample = 0
predictions = []
labels = []
exit = False
    
for it in range(30000):
    optimizer.zero_grad()
    losses = []
    corrects = []
    points_batch = []
    cutdim_batch = []
    targets = []
    bt = batch_size
    start = time.time()
    for batch in range(bt):
        #j = np.random.randint(l)
        #point_set, class_label = d[j]
        if not DATASET.has_next_batch():
            if test:
                exit = True
                break
            else:
                DATASET.reset()
                DATASET.has_next_batch()
            
        batch_data, class_label = DATASET.next_batch(augment= not test)
        point_set = np.reshape(batch_data,(num_points,3))
        cutdim, tree = make_cKDTree(point_set, depth=levels)
        targets.append(torch.tensor(class_label).long())
        
        """if batch == 0 and it ==0:
            tree = [[] for i in range(levels + 1)]
            cutdim = [[] for i in range(levels)]
            tree[0].append(point_set)

            for level in range(levels):
                for item in tree[level]:
                    left_ps, right_ps, dim = split_ps(item)
                    tree[level+1].append(left_ps)
                    tree[level+1].append(right_ps)
                    cutdim[level].append(dim)
                    cutdim[level].append(dim)

        else:
            tree[0] = [point_set]
            for level in range(levels):
                for pos, item in enumerate(tree[level]):
                    split_ps_reuse(item, level, pos, tree, cutdim)
                        #print level, pos"""

        
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
    if not test:
        loss.backward()
    losses.append(loss.data[0])
    
    if not test:
        optimizer.step()
        if (it+start_epoch) % 2000 == 0 and it!=0:
            torch.save(net.state_dict(), 'logs/model.pth-%d' % (it+start_epoch))
        end = time.time()
        print('batch: %d, loss: %f, correct %d/%d' %( it+start_epoch, np.mean(losses), correct, bt))
        print('Time: %f' % ((float(end)-start)/batch_size))
    else:
        sum_correct += correct
        sum_sample += bt
        if sum_sample > 0:
            print("accuracy: %d/%d = %f" % (sum_correct, sum_sample, float(sum_correct) / float(sum_sample)))
        predictions+=(pred_choice.tolist())
        labels+=(target_v.tolist())
    if exit:
        count_and_exit(sum_correct, sum_sample, predictions, labels)
    

    

