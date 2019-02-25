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
# ['data'], ['label'] h5py format trainfiles.txt and testfiles.txt

DATASETS = ['train', 'test', 'val']

def save_for_pnet(args, files, categories, split):
    procesor = MultiProcesor(files, args.t, args.l, categories, split, args.m, args.dataset, file_to_pointcloud, write_for_pnet)
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
    
    for dataset in DATASETS:
        with open (os.path.join(dest,"{}_files.txt").format(dataset), 'w') as f:
            for file in files:
                if file.split('_')[0] == dataset:
                    print(os.path.join(dest,file),file=f)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("d", type=str, help="root directory of dataset to be sampled")
    parser.add_argument("o", type=str, help="directory of the output files")
    
    parser.add_argument("-n", default=2048, type=int, help="Number of points to smaple")
    parser.add_argument("-t", default = 16, type=int, help="Number of threads")
    parser.add_argument("-m", default = 10000, type=int, help="Max number of models to save to one file")
    parser.add_argument("-l",default ="/data/logpnet.txt", type=str, help="logging file")
    
    parser.add_argument("--dataset",default ="shapenet", type=str, help="Dataset to convert:currently supported")

    
    args = parser.parse_args()
    args.l = os.path.join(args.o, 'log.txt')
    args.normal = False
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
    
    if not os.path.isdir(args.o):
        os.system("mkdir -m 777 {}".format(args.o))
    
    shuffle(files)
    save_for_pnet(args, files, categories, split)
    collect_files(args.o)
    