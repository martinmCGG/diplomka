from __future__ import print_function
import os
from Shapenet import get_shapenet_metadata
from Modelnet import get_modelnet_metadata
from multiprocessing import Process, Pool
from pathlib import Path
from mesh_files import *


def get_name_of_image_file(output_dir, file_id, angle):
    return os.path.join(output_dir , file_id, file_id + "_{:.2f}.png".format(angle))
    
def get_name_of_txt_file(output_dir, file_id):
    return os.path.join(output_dir , file_id, file_id + ".txt")

def render_one_image(geometry, unformated_scene, angle, output_dir, id, file_id, fov, dodecahedron = False):
    output_file = get_name_of_image_file(output_dir, file_id, angle)
    
    if dodecahedron:
        formated_scene = unformated_scene.format(output_file, geometry, 0, dodecahedron, fov)
        #print(formated_scene)
    else:
        formated_scene = unformated_scene.format(output_file, geometry, angle, "1 0.2 0")
        
    with open("formated_scene{}.pbrt".format(id), 'w') as f:
        print(formated_scene, file=f)
    cmd = "./pbrt formated_scene{}.pbrt > /dev/null".format(id)
    #cmd = "echo \"{}\" | ./pbrt".format(formated_scene)
    os.system(cmd)
    
    
def render_model(obj_file, id, file_id, views, output_dir, cat, fov, dodecahedron=False):
    geometry = os.path.join(os.path.split(obj_file)[0] , Path(obj_file).stem + ".pbrt")
    os.system("mkdir -m 777 {}".format(os.path.join(output_dir,file_id)))
    
    cmd = "./obj2pbrt {} {}".format(obj_file, geometry)
    os.system(cmd) 
    with open("scene.pbrt", 'r') as f:
        unformated_scene = f.read() 
        
    if dodecahedron:
        views = 20
        
    for view in range(views):
        if dodecahedron:
            render_one_image(geometry, unformated_scene, view*360/views, output_dir, id, file_id,fov, dodecahedron=dodecahedron[view])
        else:
            render_one_image(geometry, unformated_scene, view*360/views, output_dir, id, file_id,fov)
        
    os.system("rm {}".format(geometry))
    with open(get_name_of_txt_file(output_dir, file_id), 'w') as f:
        print(cat, file=f)
        print(views, file=f)
        for view in range(views):
            angle = view*360/views
            print(get_name_of_image_file(output_dir, file_id, angle), file=f)
       

def files_to_images(files, id, args, categories):
    views = args.v
    output_dir = args.o
    print("STARTING {}".format(id))
    for file in files:
        file_id = get_file_id(file, args.dataset)
        render_model(file,id,file_id, views, output_dir, categories[file_id],args.fov,dodecahedron=args.dodecahedron)
    print("ENDING {}".format(id))

def save_for_mvcnn(args, files, categories):
    size = len(files) // args.t
    pool = []
    if args.dodecahedron:
        args.dodecahedron = compute_dodecahedron_vertices()
    else:
        args.dodecahedron = False
        
    if len(files) > 20:
        for i in range(args.t-1):
            p = Process(target=files_to_images, args=(files[i*size:(i+1)*size], i, args, categories))
            p.start()
            pool.append(p)
        p = Process(target=files_to_images, args=(files[(args.t-1)*size:], args.t-1, args, categories))
        p.start()
        pool.append(p)
        for p in pool:
            p.join()
    else:
        files_to_images(files[i*size:(i+1)*size], 0, args, categories)

def collect_files(files, split, args):
    print("COLLECTING")
    datasets = ['train', 'test', 'val']
    for dataset in range(len(datasets)):
        with open ('{}/{}.txt'.format(args.o, datasets[dataset]), 'w') as f:
            for file in files:
                file_id = get_file_id(file, args.dataset)
                if file_id in split and split[file_id] == dataset:
                    print(get_name_of_txt_file(args.o, file_id), file = f)

def get_file_id(file, dataset):
    if dataset == "shapenet":
        return file.split('/')[-3]
    elif dataset == "modelnet":
        return file.split('/')[-1].split('.')[-2]
        
        

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("d", type=str, help="root directory of dataset to be rendered")
    parser.add_argument("o", type=str, help="directory of the output files")
    
    parser.add_argument("-v", default=12, type=int, help="Number of views to render")
    parser.add_argument("-t", default = 8, type=int, help="Number of threads")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")
    
    parser.add_argument("--dataset",default ="modelnet", type=str, help="Dataset to convert:shapenet or modelnet")
    parser.add_argument("--dodecahedron",action='store_true', help="if this is added, views will be rendered from vertices of dodecahedron")
    
    args = parser.parse_args()

    
    if args.dataset == "shapenet":
        files = find_files(args.d, 'obj')
        categories, split = get_shapenet_metadata(args.d)
        args.fov = 35
    elif args.dataset == "modelnet":
        files = find_files(args.d, 'off')
        categories, split = get_modelnet_metadata(args.d, files)
        pool = Pool(processes=args.t)
        #pool.map(read_off_file, files)
        pool.map(off2obj, files)
        files = find_files(args.d, 'obj')
        args.fov = 68
        
    if not os.path.isdir(args.o):
        os.system("mkdir -m 777 {}".format(args.o))
    save_for_mvcnn(args, files, categories)
    collect_files(files, split, args)
    
    