from __future__ import print_function
import os
import sys
import traceback
from multiprocessing import Process, Pool, Lock
from pathlib import Path
from mesh_files import *
from config import get_config, add_to_config

coding = {
    0:'train',
    1:'test',
    2:'val'
    }


def get_name_of_image_file(output_dir, angle, camera_angle, file_id, i):
    return os.path.join(output_dir, file_id + "_{:.2f}_{:.2f}.png".format(angle, camera_angle))

def get_name_of_txt_file(output_dir, cat_name, dataset, file_id):
    return os.path.join(output_dir , cat_name, dataset, file_id, file_id + ".txt")

def render_one_image(geometry, unformated_scene, angle, camera_angle, output_dir, id, file_id, fov, dodecahedron = False):

    output_file = get_name_of_image_file(output_dir, angle, camera_angle, file_id, id)
    
    if dodecahedron:
        formated_scene = unformated_scene.format(output_file, geometry, camera_angle, dodecahedron, fov, dodecahedron)
    else:
        formated_scene = unformated_scene.format(output_file, geometry, angle, "1.6 0.8 0", fov, "0 1 0")
        
    with open("formated_scene{}.pbrt".format(id), 'w') as f:
        print(formated_scene, file=f)
    cmd = "./pbrt formated_scene{}.pbrt > /dev/null".format(id)
    #cmd = "echo \"{}\" | ./pbrt".format(formated_scene)
    os.system(cmd)

    
def render_model(obj_file, id, config, cat, dataset, cat_name):
    fov = config.fov
    file_id = get_file_id(obj_file)
    geometry = os.path.join(os.path.split(obj_file)[0] , Path(obj_file).stem + ".pbrt")
    whole_path = os.path.join(config.output, cat_name, coding[dataset], file_id)
    
    os.system("mkdir -m 777 \"{}\"".format(whole_path))
  
    cmd = "./obj2pbrt {} {}".format(obj_file, geometry)
    os.system(cmd) 
    with open("scene.pbrt", 'r') as f:
        unformated_scene = f.read() 

    views = config.num_views
    if config.dodecahedron:
        views = 20

    for view in range(views):
        for camera_rotation in range(config.camera_rotations):
            if config.dodecahedron:
                render_one_image(geometry, unformated_scene, view*360/views, camera_rotation*360/config.camera_rotations,whole_path, id, file_id,fov, dodecahedron=config.d_vertices[view])
            else:
                render_one_image(geometry, unformated_scene, view*360/views, camera_rotation*360/config.camera_rotations, whole_path, id, file_id, fov)

    os.system("rm {}".format(geometry))
    
    with open(get_name_of_txt_file(config.output ,cat_name, coding[dataset], file_id), 'w') as f:
        print(cat, file=f)
        print(views, file=f)
        for view in range(views):
            angle = view*360/views
            for camera_rotation in range(config.camera_rotations):
                camera_angle = camera_rotation*360/config.camera_rotations
                print(get_name_of_image_file(whole_path, angle, camera_angle, file_id, id), file=f)

def files_to_images(files, id, config, categories, split, lock):
    views = config.num_views
    
    camera_rotations = config.camera_rotations
    log("Starting thread {} on {} files.".format(id, len(files)),lock, config.log_file)
    for i in range(len(files)):
        file = files[i]
        if i%100==0:
            log("Thread {} is {:.2f}% done.".format(id,float(i)/len(files)*100), lock, config.log_file)
        try:
            file_id = get_file_id(file)
            cat = categories[file_id]
            cat_name = config.cat_names[cat]
            dataset = split[file_id]
            render_model(file, id, config, cat, dataset, cat_name)
        except:
            e = sys.exc_info()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            log("Exception occured in thread {}. Failed to proccess file {}".format(id, file), lock, config.log_file)
            log("Exception: Type: {} File: {} Line: {}".format(exc_type, fname, exc_tb.tb_lineno), lock, config.log_file)   
    log("Ending thread {}.".format(id), lock, config.log_file)
    
    
def save_for_mvcnn(config, files, categories, split):
    size = len(files) // config.num_threads
    pool = []
    lock = Lock()
    if config.dodecahedron:
        config = add_to_config(config,'d_vertices', cat_names)
    
    for cat in config.cat_names:
        os.system("mkdir -m 777 \"{}\"".format(os.path.join(config.output,cat)))
        for dataset in coding.values():
            os.system("mkdir -m 777 \"{}\"".format(os.path.join(config.output,cat,dataset)))
    
    if len(files) > 20:
        log("Starting {} threads on {} files.".format(config.num_threads, len(files)),lock, config.log_file)
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
        files_to_images(files, 0, config, categories,split, lock)
    log("Ending...",lock, config.log_file)

def collect_files(files, split, cats, config):
    print("COLLECTING")
    datasets = ['train', 'test', 'val']
    for dataset in datasets:
        with open ('{}/{}.txt'.format(config.output, dataset), 'w') as f:
            for file in files:
                file_id = get_file_id(file)
                cat = categories[file_id]
                if coding[split[file_id]] == dataset:
                    print("{} {}".format(get_name_of_txt_file(config.output, config.cat_names[cat], dataset , file_id), cat), file = f)


def log(message, lock, log):
    lock.acquire()
    with open(log, 'a') as f:
        print(message, file = f)
    lock.release()        
        

def compute_dodecahedron_vertices():
    phi = 1.618
    vertices = []
    for i in [-1, +1]:
        for j in [-1, +1]:
            for k in [-1, +1]:
                vertices.append((i,j,k))
    for i in [-1*phi,phi]:
        for j in [-1/phi, 1/phi]:
            vertices.append((0,i,j))
            vertices.append((j,0,i))
            vertices.append((i,j,0))
    vertices = [str(x[0]) + " " + str(x[1])+" " +str(x[2]) for x in vertices]
    return vertices
               
if __name__ == '__main__':
    
    config = get_config()
    with open(config.log_file, 'w') as f:
        print("STARTING CONVERSION", file = f)
    try:
        if config.dataset_type == "shapenet":
            from Shapenet import *
        elif config.dataset_type == "modelnet":
            from Modelnet import *
            
        categories, split, cat_names = get_metadata(config.data)
        write_cat_names(config.data, config.output)
        
        if config.dataset_type == "shapenet":
            files = get_files_list(config.data, categories)
            config = add_to_config(config,'fov', 35)
        elif config.dataset_type == "modelnet":
            files = find_files(config.data, 'off')
            pool = Pool(processes=config.num_threads)
            #pool.map(off2obj, files)
            pool.close()
            pool.join()
            files = find_files(config.data, 'obj')
            config = add_to_config(config,'fov', 70)
        config = add_to_config(config,'cat_names', cat_names)
    except: 
        e = sys.exc_info()
        with open(config.log_file, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    
    save_for_mvcnn(config, files, categories, split)
    collect_files(files, split, categories, config)
    if config.dataset_type == 'modelnet' and config.remove_obj:
        os.system('find {} -name *.obj -delete'.format(config.data))
    
    