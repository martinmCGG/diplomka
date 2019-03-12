from __future__ import print_function
import numpy as np
from random import shuffle
from mesh_files import find_files
import Shapenet
import Modelnet
from mesh_to_pointcloud import file_to_pointcloud
from MultiProcesor import MultiProcesor
import h5py 
import os
import sys
from config import get_config
# ['data'], ['label'] h5py format trainfiles.txt and testfiles.txt

DATASETS = ['train', 'test', 'val']

def save_for_pnet(config, files, categories, split):
    procesor = MultiProcesor(files, config.num_threads, config.log_file, categories, split, config.dataset_type, file_to_pointcloud, write_for_pnet)
    procesor.run(config)

def write_for_pnet(buffer, buffer_cats, dataset, id, config):
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
    
if __name__ == '__main__':
  
    config = get_config()
    with open(config.log_file, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        if config.dataset_type == "shapenet":
            files = find_files(config.data, 'obj')
            categories, split = Shapenet.get_metadata(config.data)
            Shapenet.write_cat_names(config.data, config.output)
        elif config.dataset_type == "modelnet":
            files = find_files(config.data, 'off')
            categories, split= Modelnet.get_metadata(config.data, files)
            Modelnet.write_cat_names(config.data, config.outputa)
    except:
        e = sys.exc_info()
        with open(config.log_file, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    
    shuffle(files)
    save_for_pnet(config, files, categories, split)
    collect_files(config.output)
                  