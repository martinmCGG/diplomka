from __future__ import print_function
import numpy as np
import os
import math
from mesh_to_volume import mesh_to_voxel_array
from mesh_files import find_files
from Shapenet import get_shapenet_metadata
from Modelnet import get_modelnet_metadata
from MultiProcesor import MultiProcesor
from npz_join import join_npz

def write_for_vrnens(buffer, buffer_cats, dataset, id, n, args):
    features = np.array(buffer)
    features = np.reshape(features, (-1,1,args.vr, args.vr, args.vr))
    cats = np.zeros(features.shape[0])
    for i in range(len(buffer_cats)):
        for j in range(args.r):
            cats[i*args.r+j] = buffer_cats[i]
    file = os.path.join(args.o, "{}_{}_{}.npz".format(dataset,id,n))
    np.savez(file ,features=features, targets=cats)

def save_for_VRNENS(args, categories, split, files):
    procesor = MultiProcesor(files, args.t, args.l, categories, split, args.m, args.dataset, mesh_to_voxel_array, write_for_vrnens)
    procesor.run(args)
        
def create_ROT_MATRIX(rotation):
    angle = math.pi/180 * 360/rotation
    matrix = np.zeros((3,3))
    matrix[0,0] = np.cos(angle)
    matrix[0,1] = -1 * np.sin(angle)
    matrix[1,0] = np.sin(angle)
    matrix[1,1] = np.cos(angle)
    matrix[2,2] = 1
    return matrix   

def collect_files(args):
    datasets = ['val', 'train', 'test']
    with open(args.l, 'a') as f:
        print("Collecting - joining npz files.", file=f)
    for dataset in datasets:
        join_npz(args.o, "{}.*\.npz".format(dataset), os.path.join(args.o, "{}.npz".format(dataset)))
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("d", type=str, help="root directory of dataset to be voxelized")
    parser.add_argument("o", type=str, help="directory of the output files")
    
    parser.add_argument("-vr", default=32, type=int, help="Resolution of the voxel grid")
    parser.add_argument("-r", default = 24, type=int, help="Number of rotations of model along vertical axis")
    parser.add_argument("-t", default = 8, type=int, help="Number of threads")
    parser.add_argument("-m", default = 2000, type=int, help="Maximum number of models to be saved in one npz file")

    parser.add_argument("-l",default ="/data/log.txt", type=str, help="logging file")
    parser.add_argument("--dataset",default ="modelnet", type=str, help="Dataset to convert,currently supported:shapenet, modelnet")

    args = parser.parse_args()
    
    with open(args.l, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        ROT_MATRIX = create_ROT_MATRIX(args.r)
        args.matrix = ROT_MATRIX
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
    #save_for_VRNENS(args, categories, split, files)
    collect_files(args)
    
    with open(args.l, 'a') as f:
        print("Ended", file=f)


