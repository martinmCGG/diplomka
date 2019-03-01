from __future__ import print_function
import os
import sys
import Shapenet
import Modelnet
from multiprocessing import Process, Pool, Lock
from pathlib import Path
from mesh_files import *
import re
from multiprocessing import Pool
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


def convert_one_category(arguments):
    input_dir, output_dir, category_id, all_test_file, all_train_file, args = arguments
    print(input_dir, output_dir)
    os.system("mkdir -m 777 {}".format(output_dir))
    
    scanner = DirectoryTreeScanner(view_num=args.rotations, flags=False, normalize=True)
    scanner.scan_tree(input_base_folder=input_dir, output_base_folder=output_dir, num_threads=8)
    
    train_file = collect_files(output_dir, 'train')
    test_file = collect_files(output_dir, 'test')
    
    test_dir = os.path.join(output_dir, 'test')
    os.system("mkdir -m 777 {}".format(test_dir))
    train_dir = os.path.join(output_dir, 'train')
    os.system("mkdir -m 777 {}".format(train_dir))
    
    adaptive = "--adaptive 1 --node_dis 1 --depth 5" if args.adaptive else ""
    os.system('/workspace/bin/octree --filenames {} --output_path {} --rot_num 1 {}'.format(test_file, test_dir, adaptive))
    os.system('/workspace/bin/octree --filenames {} --output_path {} {}'.format(train_file, train_dir, adaptive))
    
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
    parser.add_argument("--rotations", default = 1, type=int, help="Number of threads")
    parser.add_argument("-t", default = 10, type=int, help="Number of threads")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--dataset",default ="modelnet", type=str, help="Dataset to convert:shapenet or modelnet")
    
    print("STARTING")
    args = parser.parse_args()
    
    if not os.path.isdir(args.o):    
        os.system("mkdir -m 777 {}".format(args.o))
    
    os.system("rm -rf {}/*".format(args.o))
    print("Previous data removed")
    #clean_off_folder(args.d)
    print("ModelNet cleaned")
    input_categories = sorted([os.path.join(args.d,cat) for cat in os.listdir(args.d) if os.path.isdir(os.path.join(args.d,cat))])
    output_categories = sorted([os.path.join(args.o,cat) for cat in os.listdir(args.d) if os.path.isdir(os.path.join(args.d,cat))])
    
    all_test_file = os.path.join(args.o, 'test.txt')
    all_train_file = os.path.join(args.o, 'train.txt')
    
    arguments = [(input_categories[i], output_categories[i], i, all_test_file, all_train_file, args) for i in range(0,len(input_categories))]

    pool = Pool(processes=args.t)
    pool.map(convert_one_category, arguments)
    pool.close()
    pool.join()
    
    #for i in range(len(input_categories)):
    #   convert_one_category( input_categories[i], output_categories[i], i, all_test_file, all_train_file)
    
    os.system('/opt/caffe/build/tools/convert_octree_data {} {} {}'.format("/",  os.path.join(args.o,'train.txt'), os.path.join(args.o,'train_lmdb')))
    os.system('/opt/caffe/build/tools/convert_octree_data {} {} {}'.format("/", os.path.join(args.o,'test.txt'), os.path.join(args.o,'test_lmdb')))
    
    #os.system('/opt/caffe/build/tools/upgrade_octree_database {} {}'.format(os.path.join(args.o,'_train_lmdb'), os.path.join(args.o,'train_lmdb')))
    #os.system('/opt/caffe/build/tools/upgrade_octree_database {} {}'.format(os.path.join(args.o,'_test_lmdb'), os.path.join(args.o,'test_lmdb')))
    
    