from __future__ import print_function
import os
import sys
sys.path.append("/blender_scripts")

import Shapenet
import Modelnet
from multiprocessing import Process, Pool, Lock
from pathlib import Path
from mesh_files import *
from blender_scripts import *


def get_name_of_txt_file(output_dir, file_id):
    return os.path.join(output_dir , file_id, file_id + ".txt")
    
def render_model(obj_file, id, file_id, views, output_dir, cat):
    os.system("mkdir -m 777 {}".format(os.path.join(output_dir,file_id)))
    
    render_one_model(obj_file, file_id, output_dir, nviews=views)
    
    with open(get_name_of_txt_file(output_dir, file_id), 'w') as f:
        print(cat, file=f)
        print(views, file=f)
        for view in range(views):
            print(get_name_of_image_file(output_dir, file_id, view), file=f)
       

def files_to_images(files, id, args, categories, lock):
    views = args.v
    output_dir = args.o
    log("Starting thread {} on {} files.".format(id, len(files)),lock, args.l)
    for i in range(len(files)):
        file = files[i]
        log("Thread {} is {:.2f}% done.".format(id,float(i)/len(files)*100), lock, args.l)
        try:
            file_id = get_file_id(file, args.dataset)
            render_model(file, id, file_id, views, output_dir, categories[file_id])
        except:
            e = sys.exc_info()[0]
            log("Exception occured in thread {}. Failed to proccess file {}".format(id, file), lock, args.l)
            log("Exception: {}".format(e), lock, args.l)
    log("Ending thread {}.".format(id), lock, args.l)
    
    
def save_for_mvcnn(args, files, categories):
    size = len(files) // args.t
    pool = []
    lock = Lock()
    #delete_cube()
    log("Starting {} threads on {} files.".format(args.t, len(files)),lock, args.l)
    if len(files) > 20:
        for i in range(args.t-1):
            p = Process(target=files_to_images, args=(files[i*size:(i+1)*size], i, args, categories, lock))
            p.start()
            pool.append(p)
        p = Process(target=files_to_images, args=(files[(args.t-1)*size:], args.t-1, args, categories, lock))
        p.start()
        pool.append(p)
        for p in pool:
            p.join()
    else:
        files_to_images(files, 0, args, categories, lock)
    log("Ending...",lock, args.l)

def collect_files(files, split, cats, args):
    print("COLLECTING")
    datasets = ['train', 'test', 'val']
    for dataset in range(len(datasets)):
        with open ('{}/{}.txt'.format(args.o, datasets[dataset]), 'w') as f:
            for file in files:
                file_id = get_file_id(file, args.dataset)
                cat = categories[file_id]
                if (file_id not in split and dataset=='train') or  split[file_id] == dataset:
                    print("{} {}".format(get_name_of_txt_file(args.o, file_id), cat), file = f)

def get_file_id(file, dataset):
    if dataset == "shapenet":
        return file.split('/')[-3]
    elif dataset == "modelnet":
        return file.split('/')[-1].split('.')[-2]

def log(message, lock, log):
    lock.acquire()
    with open(log, 'a') as f:
        print(message, file = f)
    lock.release()        
        
               
if __name__ == '__main__':
    
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="root directory of dataset to be rendered")
    parser.add_argument("-o", type=str, help="directory of the output files")
    
    parser.add_argument("-v", default=12, type=int, help="Number of views to render")
    parser.add_argument("-t", default = 8, type=int, help="Number of threads")
    parser.add_argument("-l",default ="/data/log.txt", type=str, help="logging file")
    parser.add_argument("--dataset",default ="modelnet", type=str, help="Dataset to convert:shapenet or modelnet")

    args, unknown = parser.parse_known_args(argv)
    
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
            pool = Pool(processes=args.t)
            #pool.map(off2obj, files)
            pool.close()
            pool.join()
            files = find_files(args.d, 'obj')
    except: 
        e = sys.exc_info()
        with open(args.l, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    
    if not os.path.isdir(args.o):
        os.system("mkdir -m 777 {}".format(args.o))

    save_for_mvcnn(args, files, categories)
    collect_files(files, split,categories, args)
    
    