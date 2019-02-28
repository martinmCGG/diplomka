from __future__ import print_function
import os
import sys
import Shapenet
import Modelnet
from multiprocessing import Process, Pool, Lock
from pathlib import Path
from mesh_files import *
import re
from ocnn.virtualscanner import VirtualScanner
from ocnn.virtualscanner import DirectoryTreeScanner

sys.path.append("/workspace/ocnn/ocnn/octree/python/ocnn/utils/")
from off_utils import clean_off_folder

def collect_files(directory, name):
    print("COLLECTING {}".format(name))
    point_file =  os.path.join(directory,'{}.txt'.format(name))
    files = find_files(directory, 'points')
    files = [file for file in files if re.match('.*{}.*'.format(name), file)]
    with open (point_file, 'w') as f:
        for file in files:
            print(file, file=f)
    return point_file


def convert_one_category(input_dir, output_dir, category_id, all_test_file, all_train_file):
    print(input_dir, output_dir)
    os.system("mkdir -m 777 {}".format(output_dir))
    
    scanner = DirectoryTreeScanner(view_num=14, flags=False, normalize=True)
    scanner.scan_tree(input_base_folder=input_dir, output_base_folder=output_dir, num_threads=16)
    
    train_file = collect_files(output_dir, 'train')
    test_file = collect_files(output_dir, 'test')
    
    test_dir = os.path.join(output_dir, 'test')
    os.system("mkdir -m 777 {}".format(test_dir))
    train_dir = os.path.join(output_dir, 'train')
    os.system("mkdir -m 777 {}".format(train_dir))
    
    os.system('/workspace/bin/octree --filenames {} --output_path {}'.format(test_file, test_dir))
    os.system('/workspace/bin/octree --filenames {} --output_path {}'.format(train_file, train_dir))
    
    train_files = [ file for file in find_files(output_dir, 'octree') if re.match('.*train.*', file)]
    test_files = [ file for file in find_files(output_dir, 'octree') if re.match('.*test.*', file)]
    
    with open(all_train_file, 'a') as f:
        for file in train_files:
            print('{} {}'.format(file, category_id), file=f)
    with open(all_test_file, 'a') as f:
        for file in test_files:
            print('{} {}'.format(file, category_id), file=f)
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("d", type=str, help="root directory of dataset to be rendered")
    parser.add_argument("o", type=str, help="directory of the output files")
    
    parser.add_argument("-n", default=2048, type=int, help="Number of views to render")
    parser.add_argument("-t", default = 8, type=int, help="Number of threads")
    
    parser.add_argument("--dataset",default ="modelnet", type=str, help="Dataset to convert:shapenet or modelnet")
    
    print("STARTING")
    args = parser.parse_args()
    
    if not os.path.isdir(args.o):    
        os.system("mkdir -m 777 {}".format(args.o))
    
    os.system("rm -rf {}/*".format(args.o))
    clean_off_folder(args.d)
    input_categories = sorted([os.path.join(args.d,cat) for cat in os.listdir(args.d) if os.path.isdir(os.path.join(args.d,cat))])
    output_categories = sorted([os.path.join(args.o,cat) for cat in os.listdir(args.d) if os.path.isdir(os.path.join(args.d,cat))])
    
    all_test_file = os.path.join(args.o, 'test.txt')
    all_train_file = os.path.join(args.o, 'train.txt')
    
    for i in range(39,len(input_categories)):
        convert_one_category( input_categories[i], output_categories[i], i, all_test_file, all_train_file)
    
    os.system('/opt/caffe/build/tools/convert_octree_data {} {} {}'.format("/",  os.path.join(args.o,'train.txt'), os.path.join(args.o,'train_lmdb')))
    os.system('/opt/caffe/build/tools/convert_octree_data {} {} {}'.format("/", os.path.join(args.o,'test.txt'), os.path.join(args.o,'test_lmdb')))
    
    
    