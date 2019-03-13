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
from config import get_config
sys.path.append("/workspace/ocnn/ocnn/octree/python/ocnn/utils/")
from off_utils import clean_off_folder


coding = {
    0:'train',
    1:'test',
    2:'val'
    }

def collect_files(directory, name, dataset_type = 'modelnet', split = None):
    print("COLLECTING {}".format(name))
    point_file =  os.path.join(directory,'{}.txt'.format(name))
    files = find_files(directory, 'points')
    if dataset_type == 'modelnet':
        files = [file for file in files if re.match('.*{}.*'.format(name), file)]
    else:
        files = [file for file in files if coding[split[file.split('/')[-3]]] == name]
    
    print(files)
    with open (point_file, 'w') as f:
        for file in files:
            print(file, file=f)
    return point_file

def convert_one_category(arguments):
    input_dir, output_dir, category_id, all_test_file, all_train_file, dataset_type, adaptive, num_rotations, split = arguments

    os.system("mkdir -m 777 {}".format(output_dir))
    
    scanner = DirectoryTreeScanner(view_num=num_rotations, flags=False, normalize=True)
    scanner.scan_tree(input_base_folder=input_dir, output_base_folder=output_dir, num_threads=2)
    
    train_file = collect_files(output_dir, 'train', dataset_type, split = split)
    test_file = collect_files(output_dir, 'test', dataset_type, split = split)
    
    test_dir = os.path.join(output_dir, 'test')
    os.system("mkdir -m 777 {}".format(test_dir))
    train_dir = os.path.join(output_dir, 'train')
    os.system("mkdir -m 777 {}".format(train_dir))
    
    adaptive = "--adaptive 1 --node_dis 1 --depth 5" if adaptive else ""
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
    print("STARTING")
    config = get_config()
    with open(config.log_file, 'w') as f:
        print("STARTING CONVERSION", file = f)
        os.system("rm -rf {}/*".format(config.output))
        print("previous data removed", file = f)
    if config.clean_off_files:
        clean_off_folder(config.data)
        print(".off files cleaned", file = f)    
    try:
        if config.dataset_type == "shapenet":
            #files = find_files(config.data, 'obj')
            categories, split = Shapenet.get_metadata(config.data)
            cat_names = Shapenet.get_cat_names(config.data)
            input_categories = sorted([os.path.join(config.data,cat) for cat in os.listdir(config.data) if os.path.isdir(os.path.join(config.data,cat))])
            output_categories = sorted([os.path.join(config.output,cat) for cat in cat_names]) 
            Shapenet.write_cat_names(config.data, config.output)

        elif config.dataset_type == "modelnet":
            #files = find_files(config.data, 'off')
            #categories, split= Modelnet.get_metadata(config.data, files)
            split = None
            cat_names = Modelnet.get_cat_names(config.data)
            Modelnet.write_cat_names(config.data, config.output)
            input_categories = sorted([os.path.join(config.data,cat) for cat in cat_names])
            output_categories = sorted([os.path.join(config.output,cat) for cat in cat_names])       
    except: 
        e = sys.exc_info()
        with open(config.log_file, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
        
    all_test_file = os.path.join(config.output, 'test.txt')
    all_train_file = os.path.join(config.output, 'train.txt')
    
    arguments = [(input_categories[i], output_categories[i], i, all_test_file, all_train_file, config.dataset_type, config.adaptive, config.num_rotations, split) for i in range(0,len(input_categories))]

    pool = Pool(processes=config.num_threads)
    pool.map(convert_one_category, arguments)
    pool.close()
    pool.join()
    
    os.system('/opt/caffe/build/tools/convert_octree_data {} {} {}'.format("/",  os.path.join(config.output,'train.txt'), os.path.join(config.output,'train_lmdb')))
    os.system('/opt/caffe/build/tools/convert_octree_data {} {} {}'.format("/", os.path.join(config.output,'test.txt'), os.path.join(config.output,'test_lmdb')))
    
    
    