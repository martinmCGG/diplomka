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
from config import get_config, add_to_config

coding = {
    0:'train',
    1:'test',
    2:'val'
    }

def get_name_of_txt_file(output_dir,cat, dataset, file_id):
    return os.path.join(output_dir ,cat,dataset, file_id, file_id + ".txt")
    
def render_model(obj_file, id, file_id, views, output_dir, cat):
    os.system("mkdir -m 777 {}".format(os.path.join(output_dir,file_id)))
    
    render_one_model(obj_file, file_id, output_dir, nviews=views)
    
    with open(get_name_of_txt_file(output_dir, file_id), 'w') as f:
        print(cat, file=f)
        print(views, file=f)
        for view in range(views):
            print(get_name_of_image_file(output_dir, file_id, view), file=f)
       

def files_to_images(files, id, config, categories, lock):
    views = config.num_views
    output_dir = config.output
    log("Starting thread {} on {} files.".format(id, len(files)),lock, config.log_file)
    for i in range(len(files)):
        file = files[i]
        if i%100 == 0:
            log("Thread {} is {:.2f}% done.".format(id,float(i)/len(files)*100), lock, config.log_file)
        try:
            file_id = get_file_id(file, config.dataset_type)
            output_dir = os.path.join(config.output, config.cat_names[categories[file_id]], coding[split[file_id]])
            render_model(file, id, file_id, views, output_dir, categories[file_id])
        except:
            e = sys.exc_info()[0]
            log("Exception occured in thread {}. Failed to proccess file {}".format(id, file), lock, config.log_file)
            log("Exception: {}".format(e), lock, config.log_file)
    log("Ending thread {}.".format(id), lock, config.log_file)
    
    
def save_for_mvcnn(config, files, categories, split):
    size = len(files) // config.num_threads
    pool = []
    lock = Lock()
    
    for cat in config.cat_names:
        os.system("mkdir -m 777 {}".format(os.path.join(config.output,cat)))
        for dataset in coding.values():
            os.system("mkdir -m 777 {}".format(os.path.join(config.output,cat,dataset)))
            
    log("Starting {} threads on {} files.".format(config.num_threads, len(files)),lock, config.log_file)
    if len(files) > 20:
        for i in range(config.num_threads-1):
            p = Process(target=files_to_images, args=(files[i*size:(i+1)*size], i, config, categories, split, lock))
            p.start()
            pool.append(p)
        p = Process(target=files_to_images, args=(files[(config.num_threads-1)*size:], config.num_threads-1, config, categories,split, lock))
        p.start()
        pool.append(p)
        for p in pool:
            p.join()
    else:
        files_to_images(files, 0, config, categories, lock)
    log("Ending...",lock, config.log_file)

def collect_files(files, split, cats, config):
    print("COLLECTING")
    datasets = ['train', 'test', 'val']
    for dataset in datasets:
        with open ('{}/{}.txt'.format(config.output, dataset), 'w') as f:
            for file in files:
                file_id = get_file_id(file, config.dataset_type)
                cat = categories[file_id]
                if (file_id not in split and dataset=='train') or  coding[split[file_id]] == dataset:
                    print("{} {}".format(get_name_of_txt_file(config.output, config.cat_names[split[file_id]] ,dataset , file_id), cat), file = f)


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
    
    config = get_config()
    
    with open(config.log_file, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        if config.dataset_type == "shapenet":
            files = find_files(config.data, 'obj')
            categories, split = Shapenet.get_metadata(config.data)
            cat_names = Shapenet.get_cat_names(config.data)
            Shapenet.write_cat_names(config.data, config.output)
        elif config.dataset_type == "modelnet":
            files = find_files(config.data, 'off')
            categories, split= Modelnet.get_metadata(config.data, files)
            Modelnet.write_cat_names(config.data, config.output)
            cat_names = Modelnet.get_cat_names(config.data)
            pool = Pool(processes=config.num_threads)
            pool.map(off2obj, files)
            pool.close()
            pool.join()
            files = find_files(config.data, 'obj')
        config = add_to_config(config,'cat_names', cat_names)
    except: 
        e = sys.exc_info()
        with open(config.log_file, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    print('here')
    save_for_mvcnn(config, files, categories,split)
    collect_files(files, split,categories, config)
    if config.dataset_type and config.remove_obj:
        os.system('find {} -name *.obj -delete'.format(config.data))
    
    