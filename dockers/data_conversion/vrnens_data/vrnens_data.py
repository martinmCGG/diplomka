from __future__ import print_function
import numpy as np
import os
import sys
import math
from mesh_to_volume import mesh_to_voxel_array
from mesh_files import find_files
import Shapenet
import Modelnet

from MultiProcesor import MultiProcesor
from npz_join import join_npz
from config import get_config, add_to_config

def write_for_vrnens(buffer, buffer_cats, dataset, id, config):
    features = np.array(buffer)
    vr = config.num_voxels
    features = np.reshape(features, (-1,1,vr, vr, vr))
    cats = np.zeros(features.shape[0])
    for i in range(len(buffer_cats)):
        for j in range(config.num_rotations):
            cats[i*config.num_rotations+j] = buffer_cats[i]
    file = os.path.join(config.output, "{}_{}.npz".format(dataset,id))
    np.savez(file ,features=features, targets=cats)

def save_for_VRNENS(config, categories, split, files):
    procesor = MultiProcesor(files, config.num_threads, config.log_file, categories, split, config.dataset_type, mesh_to_voxel_array, write_for_vrnens)
    procesor.run(config)
        
def create_ROT_MATRIX(rotation):
    angle = math.pi/180 * 360/rotation
    matrix = np.zeros((3,3))
    matrix[0,0] = np.cos(angle)
    matrix[0,1] = -1 * np.sin(angle)
    matrix[1,0] = np.sin(angle)
    matrix[1,1] = np.cos(angle)
    matrix[2,2] = 1
    return matrix   

def collect_files(config):
    datasets = ['val', 'train', 'test']
    with open(config.log_file, 'a') as f:
        print("Collecting - joining npz files.", file=f)
    for dataset in datasets:
        join_npz(config.output, "{}.*\.npz".format(dataset), os.path.join(config.output, "{}.npz".format(dataset)))
        

if __name__ == '__main__':
    config = get_config()
    with open(config.log_file, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        ROT_MATRIX = create_ROT_MATRIX(config.num_rotations)
        config = add_to_config(config,'matrix', ROT_MATRIX)

        if config.dataset_type == "shapenet":
            files = find_files(config.data, 'obj')
            categories, split = Shapenet.get_metadata(config.data)
            Shapenet.write_cat_names(config.data, config.output)
        elif config.dataset_type == "modelnet":
            files = find_files(config.data, 'off')
            categories, split= Modelnet.get_metadata(config.data, files)
            Modelnet.write_cat_names(config.data, config.data)
    except:
        e = sys.exc_info()
        with open(config.log_file, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    

    save_for_VRNENS(config, categories, split, files)
    collect_files(config)
    
    with open(config.log_file, 'a') as f:
        print("Ended", file=f)


