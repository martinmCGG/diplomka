from __future__ import print_function

import os
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision

from random import shuffle
from mesh_files import find_files
from Shapenet import get_shapenet_metadata
from Modelnet import get_modelnet_metadata
from mesh_to_pointcloud import file_to_pointcloud
from MultiProcesor import MultiProcesor

import sys
sys.path.append('/sonet/util')
from som import *

# ['data'], ['label'] h5py format trainfiles.txt and testfiles.txt



DATASETS = ['train', 'test', 'val']

def save_for_sonet(args, files, categories, split):
    procesor = MultiProcesor(files, args.t, args.l, categories, split, args.m, args.dataset, file_to_pointcloud, write_for_sonet)
    procesor.run(args)

def write_for_sonet(buffer, buffer_cats, dataset, id, n, args):
    features = np.array(buffer)
    cats =  np.array(buffer_cats)
    file = os.path.join(args.o, "{}_{}_{}.npz".format(dataset,id,n))
    np.savez(file,data=features, label=cats)
    
def collect_files(dest):
    files = os.listdir(dest)
    files = [file for file in files if file.split('.')[-1] == 'npz']
    
    for dataset in DATASETS:
        with open (os.path.join(dest,"{}_files.txt").format(dataset), 'w') as f:
            for file in files:
                if file.split('_')[0] == dataset:
                    print(os.path.join(dest,file),file=f)

def add_soms(dest):
    files = os.listdir(dest)
    files = [file for file in files if file.split('.')[-1] == 'npz']
    for file in files:
        add_soms_to_file(os.path.join(dest,file))
        
def add_soms_to_file(file):

    som_builder = SOM(8, 8, 3, True)
    with np.load(file) as data:
        features = data['data']
        labels = data['label']
    som_nodes = []
    print(features, labels)
    for pc in features:
        som_nodes.append(som_cloud(pc[:, 0:3], som_builder))
    np.savez(file, data=features, label=labels, som=som_nodes)

def som_cloud(pc_np, som_builder):
    pc_np_sampled = pc_np[np.random.choice(pc_np.shape[0], min(4096,pc_np.shape[0]), replace=False), :]
    pc = torch.from_numpy(pc_np_sampled.transpose().astype(np.float32)).cuda()  # 3xN tensor
    som_builder.optimize(pc)
    som_node_np = som_builder.node.cpu().numpy().transpose().astype(np.float32)  # node_numx3
    return som_node_np
   
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("d", type=str, help="root directory of dataset to be sampled")
    parser.add_argument("o", type=str, help="directory of the output files")
    
    parser.add_argument("-n", default=2048, type=int, help="Number of points to smaple")
    parser.add_argument("-t", default = 8, type=int, help="Number of threads")
    parser.add_argument("-m", default = 10000, type=int, help="Max number of models to save to one file")
    parser.add_argument("-l",default ="/data/log.txt", type=str, help="logging file")
    
    parser.add_argument("--dataset",default ="modelnet", type=str, help="Dataset to convert:currently supported")
    parser.add_argument("--normal",action='store_true', help="if normal information should be saved")
        
    args = parser.parse_args()
    args.normal = True
    
    with open(args.l, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        if args.dataset == "shapenet":
            files = find_files(args.d, 'obj')
            categories, split = get_shapenet_metadata(args.d)
        elif args.dataset == "modelnet":
            files = find_files(args.d, 'off')
            categories, split = get_modelnet_metadata(args.d, files)
    except:
        e = sys.exc_info()
        with open(args.l, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    
    if not os.path.isdir(args.o):
        os.system("mkdir -m 777 {}".format(args.o))
    
    shuffle(files)
    save_for_sonet(args, files, categories, split)
    collect_files(args.o)
    add_soms(args.o)
    