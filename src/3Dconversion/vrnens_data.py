from __future__ import print_function
import numpy as np
import os
import math
from mesh_to_volume import obj_to_voxel_array
from obj_files import find_files
from Shapenet import get_shapenet_metadata
from MultiProcesor import MultiProcesor

def write_for_vrnens(buffer, buffer_cats,dataset, id, n, args):
    features = np.array(buffer)
    features = np.reshape(features, (-1,1,args.v, args.v, args.v))
    cats = np.zeros(features.shape[0])
    for i in range(len(buffer_cats)):
        for j in range(args.r):
            cats[i*args.r+j] = buffer_cats[i]
    print(features.shape, cats.shape)
    file = os.path.join(args.o, "{}_{}_{}.npz".format(dataset,id,n))
    np.savez(file ,features=features, targets=cats)

def save_for_VRNENS(args, categories, split):
    files = find_files(args.d, 'obj')
    procesor = MultiProcesor(files, args.t, args.l, categories, split, args.m, obj_to_voxel_array, write_for_vrnens)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=32, type=int, help="Resolution of the voxel grid")
    parser.add_argument("-d", type=str, help="root directory of .obj files to be voxelizes")
    parser.add_argument("-r", default = 24, type=int, help="Number of rotations of model along vertical axis")
    parser.add_argument("-t", default = 8, type=int, help="Number of threads")
    parser.add_argument("-m", default = 2000, type=int, help="Maximum number of models to be saved in one npz file")
    parser.add_argument("-o", type=str, help="directory of the output files")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")

    args = parser.parse_args()
    
    ROT_MATRIX = create_ROT_MATRIX(args.r)
    args.matrix = ROT_MATRIX
    categories, split = get_shapenet_metadata(args.d)
    
    save_for_VRNENS(args, categories, split)

