
from __future__ import print_function
import numpy as np
from obj_files import find_files
from Shapenet import get_shapenet_metadata
from mesh_to_pointcloud import obj_to_pointcloud
from MultiProcesor import MultiProcesor
import h5py 
import os
# ['data'], ['label'] h5py format trainfiles.txt and testfiles.txt

def save_for_pnet(args, categories, split):
    files = find_files(args.d, 'obj')
    procesor = MultiProcesor(files, args.t, args.l, categories, split, args.m, obj_to_pointcloud, write_for_pnet)
    procesor.run(args)

def write_for_pnet(buffer, buffer_cats, dataset, id, n, args):
    h5f = h5py.File(os.path.join(args.o,"{}_{}_{}.h5".format(dataset,id,n)), 'w')
    features = np.array(buffer)
    targets = np.array(buffer_cats)
    h5f.create_dataset('data', data=features)
    h5f.create_dataset('label', data=targets)
    h5f.close()

def collect_files(dest):
    files = os.listdir(dest)
    files = [file for file in files if file.split('.')[-1] == 'h5']
    datasets = ['train', 'test', 'val']
    for dataset in datasets:
        with open (os.path.join(dest,"{}files.txt").format(dataset), 'w') as f:
            for file in files:
                if file.split('_')[0] == dataset:
                    print(os.path.join(dest,file),file=f)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=2048, type=int, help="Number of points to smaple")
    parser.add_argument("-d", type=str, help="root directory of .obj files to be voxelizes")
    parser.add_argument("-t", default = 4, type=int, help="Number of threads")
    parser.add_argument("-m", default = 1000, type=int, help="Max number of models to save to one file")
    parser.add_argument("-o", type=str, help="directory of the output files")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")
        
    args = parser.parse_args()
    categories, split = get_shapenet_metadata(args.d)
    
    save_for_pnet(args, categories, split)
    collect_files(args.o)
    
    
    