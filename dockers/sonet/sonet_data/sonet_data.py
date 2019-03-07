from __future__ import print_function

import os
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import Shapenet
import Modelnet
import torch
import torchvision
import h5py 
from random import shuffle
from mesh_files import find_files
from mesh_to_pointcloud import file_to_pointcloud
from MultiProcesor import MultiProcesor


GPU = 0

import sys
os.system('CUDA_VISIBLE_DEVICES={}'.format(GPU))
sys.path.append('/sonet/util')
from som import *

# ['data'], ['label'] h5py format trainfiles.txt and testfiles.txt

DATASETS = ['train', 'test', 'val']

def save_for_sonet(args, files, categories, split):
    procesor = MultiProcesor(files, args.t, args.l, categories, split, args.m, args.dataset, file_to_pointcloud, write_for_sonet)
    procesor.run(args)
    

def write_for_sonet(buffer, buffer_cats, dataset, id, n, args):
    h5f = h5py.File(os.path.join(args.o,"{}_{}_{}.h5".format(dataset,id,n)), 'w')
    features = np.array(buffer)
    targets = np.array(buffer_cats)
    h5f.create_dataset('data', data=features)
    h5f.create_dataset('label', data=targets)
    h5f.close()
    

def collect_files(dest):
    files = os.listdir(dest)
    files = [file for file in files if file.split('.')[-1] == 'h5']
    
    for dataset in DATASETS:
        with open (os.path.join(dest,"{}_files.txt").format(dataset), 'w') as f:
            for file in files:
                if file.split('_')[0] == dataset:
                    print(os.path.join(dest,file),file=f)    
    
def add_soms(dest):
    som_builder = SOM(8, 8, 3, GPU)
    for dataset in DATASETS:
        with open (os.path.join(dest,"{}_files.txt").format(dataset), 'r') as f:
            for line in f:
                print("Somatazing file: ".format(line.strip()))
                h5file = h5py.File(line.strip(), 'r')
                data = h5file['data'][:]
                labels = h5file['label'][:]
                soms = []
                for dato in data:
                    soms.append(som_one_cloud(dato, som_builder))
                h5file.close()
                
                h5file = h5py.File(line.strip(), 'w')    
                h5file.create_dataset('data', data=data)
                h5file.create_dataset('label', data=labels)
                h5file.create_dataset('som', data=soms)                
                h5file.close()
                
                
    
def som_one_cloud(data, som_builder):
    pc_np = data[:, 0:3]
    sn_np = data[:, 3:6]
    pc_np = pc_np[np.random.choice(pc_np.shape[0], 2048, replace=False), :]
    pc = torch.from_numpy(pc_np.transpose().astype(np.float32)).cuda()  # 3xN tensor
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

        
    args = parser.parse_args()
    
    with open(args.l, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        if args.dataset == "shapenet":
            files = find_files(args.d, 'obj')
            categories, split = Shapenet.get_metadata(args.d)
            Shapenet.write_cat_names(args.d, args.o)
        elif args.dataset == "modelnet":
            files = find_files(args.d, 'off')
            categories, split= Modelnet.get_metadata(args.d, files)
            Modelnet.write_cat_names(args.d, args.d)
    except:
        e = sys.exc_info()
        with open(args.l, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    
    if not os.path.isdir(args.  o):
        os.system("mkdir -m 777 {}".format(args.o))
    print("cuda is available",torch.cuda.is_available())
    #shuffle(files)
    #save_for_sonet(args, files, categories, split)
    #collect_files(args.o)
    print("Starting soma")
    add_soms(args.o)
    
