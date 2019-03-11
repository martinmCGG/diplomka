from __future__ import print_function
import os
import sys
import Shapenet
import Modelnet
from multiprocessing import Process, Pool, Lock
from pathlib import Path
from mesh_files import *
from config import get_config, add_to_config

def get_name_of_image_file(output_dir, file_id, angle, camera_angle):
    return os.path.join(output_dir , file_id, file_id + "_{:.2f}_{:.2f}.png".format(angle, camera_angle))
    
def get_name_of_txt_file(output_dir, file_id):
    return os.path.join(output_dir , file_id, file_id + ".txt")

def render_one_image(geometry, unformated_scene, angle, camera_angle, output_dir, id, file_id, fov, dodecahedron = False):
    output_file = get_name_of_image_file(output_dir, file_id, angle, camera_angle)
    
    if dodecahedron:
        formated_scene = unformated_scene.format(output_file, geometry, camera_angle, dodecahedron, fov, dodecahedron)
    else:
        formated_scene = unformated_scene.format(output_file, geometry, angle, "1.6 0.8 0", fov, "0 1 0")
        
    with open("formated_scene{}.pbrt".format(id), 'w') as f:
        print(formated_scene, file=f)
    cmd = "./pbrt formated_scene{}.pbrt > /dev/null".format(id)
    #cmd = "echo \"{}\" | ./pbrt".format(formated_scene)
    os.system(cmd)
    
    
def render_model(obj_file, id, file_id, views, camera_rotations, output_dir, cat, fov, dodecahedron=False):
    geometry = os.path.join(os.path.split(obj_file)[0] , Path(obj_file).stem + ".pbrt")
    os.system("mkdir -m 777 {}".format(os.path.join(output_dir,file_id)))
    
    cmd = "./obj2pbrt {} {}".format(obj_file, geometry)
    os.system(cmd) 
    with open("scene.pbrt", 'r') as f:
        unformated_scene = f.read() 
        
    if dodecahedron:
        views = 20
        
    for view in range(views):
        for camera_rotation in range(camera_rotations):
            if dodecahedron:
                render_one_image(geometry, unformated_scene, view*360/views, camera_rotation*360/camera_rotations, output_dir, id, file_id,fov, dodecahedron=dodecahedron[view])
            else:
                render_one_image(geometry, unformated_scene, view*360/views, camera_rotation*360/camera_rotations, output_dir, id, file_id,fov)
        
    os.system("rm {}".format(geometry))
    with open(get_name_of_txt_file(output_dir, file_id), 'w') as f:
        print(cat, file=f)
        print(views, file=f)
        for view in range(views):
            angle = view*360/views
            for camera_rotation in range(camera_rotations):
                camera_angle = camera_rotation*360/camera_rotations
                print(get_name_of_image_file(output_dir, file_id, angle, camera_angle), file=f)
       

def files_to_images(files, id, config, categories, lock):
    views = config.num_views
    output_dir = config.output
    camera_rotations = config.camera_rotations
    log("Starting thread {} on {} files.".format(id, len(files)),lock, config.log_file)
    for i in range(len(files)):
        file = files[i]
        log("Thread {} is {:.2f}% done.".format(id,float(i)/len(files)*100), lock, config.log_file)
        try:
            file_id = get_file_id(file, config.dataset_type)
            render_model(file, id, file_id, views, camera_rotations, output_dir, categories[file_id],config.fov,dodecahedron=config.dodecahedron)
        except:
            e = sys.exc_info()[0]
            log("Exception occured in thread {}. Failed to proccess file {}".format(id, file), lock, config.log_file)
            log("Exception: {}".format(e), lock, config.log_file)
    log("Ending thread {}.".format(id), lock, config.log_file)
    
    

def save_for_mvcnn(config, files, categories):
    size = len(files) // config.num_threads
    pool = []
    lock = Lock()
    if config.dodecahedron:
        config.dodecahedron = compute_dodecahedron_vertices()
    
    if len(files) > 20:
        log("Starting {} threads on {} files.".format(config.num_threads, len(files)),lock, config.log_file)
        for i in range(config.num_threads-1):
            p = Process(target=files_to_images, config=(files[i*size:(i+1)*size], i, config, categories, lock))
            p.start()
            pool.append(p)
        p = Process(target=files_to_images, config=(files[(config.num_threads-1)*size:], config.num_threads-1, config, categories, lock))
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
    for dataset in range(len(datasets)):
        with open ('{}/{}.txt'.format(config.output, datasets[dataset]), 'w') as f:
            for file in files:
                file_id = get_file_id(file, config.dataset_type)
                cat = categories[file_id]
                if (file_id not in split and dataset=='train') or  split[file_id] == dataset:
                    print("{} {}".format(get_name_of_txt_file(config.output, file_id), cat), file = f)

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
            files = find_files(config.data, 'obj')
            categories, split = Shapenet.get_metadata(config.data)
            Shapenet.write_cat_names(config.data, config.output)
            config = add_to_config(config,'fov', 35)
        elif config.dataset_type == "modelnet":
            files = find_files(config.data, 'off')
            categories, split= Modelnet.get_metadata(config.data, files)
            Modelnet.write_cat_names(config.data, config.data)
            pool = Pool(processes=config.num_threads)
            pool.map(off2obj, files)
            pool.close()
            pool.join()
            files = find_files(config.data, 'obj')
            config = add_to_config(config,'fov', 70)
    except: 
        e = sys.exc_info()
        with open(config.log_file, 'a') as f:
            print("Exception occured while reading files.", file=f)
            print("Exception {}".format(e), file=f)
        sys.exit(1)
    
    save_for_mvcnn(config, files, categories)
    collect_files(files, split,categories, config)
    if config.dataset_type and config.remove_obj:
        os.system('find {} -name *.obj -delete'.format(config.data))
    
    