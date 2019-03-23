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
from config import get_config

GPU = 0

import sys
sys.path.append('/sonet/util')
from som import *


DATASETS = ['train', 'test', 'val']

def save_for_sonet(config, files, categories, split):
    procesor = MultiProcesor(files, config.num_threads, config.log_file, categories, split, config.dataset_type, file_to_pointcloud, write_for_sonet)
    procesor.run(config)
    

def write_for_sonet(buffer, buffer_cats, dataset, id, config):
    h5f = h5py.File(os.path.join(config.output,"{}_{}.h5".format(dataset,id)), 'w')
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
    
def add_soms(config):
    dest = config.output
    size = int(math.sqrt(config.node_num))
    som_builder = SOM(size, size, 3, GPU)
    for dataset in DATASETS:
        with open (os.path.join(dest,"{}_files.txt").format(dataset), 'r') as f:
            for line in f:
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
    
    config = get_config()
    with open(config.log_file, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        if config.dataset_type == "shapenet":
            from Shapenet import *
        elif config.dataset_type == "modelnet":
            from Modelnet import *
        
        categories, split, cat_names = get_metadata(config.data)
        files = get_files_list(config.data, categories)
        write_cat_names(config.data, config.output)
    except:
        e = sys.exc_info()
        with open(config.log_file, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    
    if not os.path.isdir(config.output):
        os.system("mkdir -m 777 {}".format(config.output))
    print("cuda is available",torch.cuda.is_available())
    shuffle(files)
    save_for_sonet(config, files, categories, split)
    collect_files(config.output)
    print("Starting soma")
    add_soms(config.output)
    
