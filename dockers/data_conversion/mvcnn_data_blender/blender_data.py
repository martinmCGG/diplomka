from __future__ import print_function
import os
import sys
import traceback
sys.path.append("/blender_scripts")
from multiprocessing import Process, Pool

from mesh_files import *
from blender_scripts import *
from config import get_config, add_to_config

coding = {
    0:'train',
    1:'test'
    }

def get_name_of_txt_file(output_dir, cat_name, dataset, file_id):
    return os.path.join(output_dir , cat_name, dataset, file_id, file_id + ".txt")
    
def render_model(obj_file, file_id, views, output_dir, cat, dataset, cat_name):
    
    whole_path = os.path.join(output_dir, cat_name, dataset, file_id)
    os.system("mkdir -p -m 777 \"{}\"".format(whole_path))
    if config.blender_script == 'phong':
        render_phong(obj_file, file_id, whole_path, nviews=views)
    else:
        render_one_model(obj_file, file_id, whole_path, nviews=views)
    
    with open(get_name_of_txt_file(output_dir ,cat_name, dataset, file_id), 'w') as f:
        print(cat, file=f)
        print(views, file=f)
        for view in range(views):
            print(get_name_of_image_file(whole_path, file_id, view), file=f)
       

def files_to_images(files, config, categories, split):
    views = config.num_views
    output_dir = config.output
    for i in range(len(files)):
        try:
            file = files[i]
            file_id = get_file_id(file)
            cat = categories[file_id]
            cat_name = config.cat_names[cat]
            render_model(file, file_id, views, output_dir, cat, coding[split[file_id]], cat_name)
        except:
            err_string = traceback.format_exc()
            log(config.log_file, "Exception occured while rendering file {}".format(file))
            log(config.log_file, err_string)   
            sys.exit(1)

    
def run_multithread(files, config, categories, split, size_thread):
    pool = []
    for i in range(config.num_threads):
        p = Process(target=files_to_images, args=(files[i* size_thread:(i+1)* size_thread], config, categories, split))
        p.start()
        pool.append(p)
    for p in pool:
        p.join()

    
def save_for_mvcnn(config, all_files, categories, split):
            
    log(config.log_file,"Starting {} threads on {} files.".format(config.num_threads, len(all_files)))
    size_thread = 100
    size  = size_thread * config.num_threads
    if len(all_files) > size_thread:
        for j in range(len(all_files)//size):
            files = all_files[j*size:(j+1)*size]
            run_multithread(files, config, categories, split, size_thread)
            log(config.log_file, "Finished {} %".format(j*size/len(all_files))) 
        files = all_files[len(all_files)//size*size:]
        run_multithread(files, config, categories, split, size_thread)
    else:
        files_to_images(all_files, config, categories,split)
    log(config.log_file, "Finished conversion")


                            
def collect_files(files, split, cats, config):
    log(config.log_file, "COLLECTING")
    datasets = ['train', 'test']
    for dataset in datasets:
        with open ('{}/{}.txt'.format(config.output, dataset), 'w') as f:
            for file in files:
                file_id = get_file_id(file)
                cat = categories[file_id]
                if coding[split[file_id]] == dataset:
                    print("{} {}".format(get_name_of_txt_file(config.output, config.cat_names[cat] , dataset , file_id), cat), file = f)                           


def log(file, log_string):
    with open(file, 'a') as f:
        print(log_string)
        print(log_string, file=f)        
               
if __name__ == '__main__':

    config = get_config()

    with open(config.log_file, 'w') as f:
        print("STARTING CONVERSION", file = f)
        print("STARTING CONVERSION")
    try:
        if config.dataset_type == "shapenet":
            from Shapenet import *
        elif config.dataset_type == "modelnet":
            from Modelnet import *
            
        categories, split, cat_names = get_metadata(config.data)
        write_cat_names(config.data, config.output)
        
        if config.dataset_type == "shapenet":
            files = get_files_list(config.data, categories)
        elif config.dataset_type == "modelnet":
            files = find_files(config.data, 'off')
            pool = Pool(processes=config.num_threads)
            log(config.log_file, "Converting off files to obj. May take a while.")
            pool.map(off2obj, files)
            pool.close()
            pool.join()
            files = find_files(config.data, 'obj')
        config = add_to_config(config,'cat_names', cat_names)
    except:
        err_string = traceback.format_exc()
        log(config.log_file, "Exception occured while reading files.")
        log(config.log_file, err_string)   
        sys.exit(1)
    
    
    def exists(file):
        id = get_file_id(file)
        cat = categories[id]
        cat_name = cat_names[cat]
        dataset = coding[split[id]]
        whole_path = os.path.join(config.output, cat_name, dataset, id)
        if not os.path.exists(os.path.join(whole_path, id + '.txt')):
            return False
        for view in range(config.num_views):
            if not os.path.exists(get_name_of_image_file(whole_path, id, view)):
                return False
        return True
    
    #Do not convert already existing files
    all_files = files
    files = [x for x in files if not exists(x)]
    
    save_for_mvcnn(config, files, categories, split)
    collect_files(all_files, split,categories, config)
    log(config.log_file, "Ending and cleaning")
    if config.dataset_type == 'modelnet' and config.remove_obj:
        log(config.log_file, "Removing .obj files.")
        os.system('find {} -name *.obj -delete'.format(config.data))
    log(config.log_file, "ENDING")
    
        
    
    