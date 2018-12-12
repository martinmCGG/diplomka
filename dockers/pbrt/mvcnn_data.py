from __future__ import print_function
import os
from obj_files import find_files
from Shapenet import get_shapenet_metadata
from Modelnet import get_modelnet_metadata
from multiprocessing import Process, Lock
from pathlib import Path

def get_name_of_image_file(output_dir, file_id, angle):
    return os.path.join(output_dir , file_id, file_id + "{:.2f}.png".format(angle))
    
def get_name_of_txt_file(output_dir, file_id):
    return os.path.join(output_dir , file_id, file_id + ".txt")

def render_one_image(geometry, unformated_scene, angle, output_dir, id, file_id):
    output_file = get_name_of_image_file(output_dir, file_id, angle)
    #print(output_file)
    #img_file = "/data/example{}.png".format(angle)
    
    formated_scene = unformated_scene.format(output_file, geometry, angle)
    with open("formated_scene{}.pbrt".format(id), 'w') as f:
        print(formated_scene, file=f)
    cmd = "./pbrt formated_scene{}.pbrt > /dev/null".format(id)
    #cmd = "echo \"{}\" | ./pbrt".format(formated_scene)
    os.system(cmd)
    
    
def render_model(obj_file, id, views, output_dir, categories):
    geometry = os.path.join(os.path.split(obj_file)[0] , Path(obj_file).stem + ".pbrt")
    file_id = obj_file.split('/')[-3]
    os.system("mkdir -m 777 {}".format(os.path.join(output_dir,file_id)))
    cat = find_category(obj_file, categories)
    cmd = "./obj2pbrt {} {}".format(obj_file, geometry)
    os.system(cmd) 
    with open("scene.pbrt", 'r') as f:
        unformated_scene = f.read()
    for view in range(views):
        render_one_image(geometry, unformated_scene, view*360/views, output_dir, id, file_id)
    os.system("rm {}".format(geometry))
    with open(get_name_of_txt_file(output_dir, file_id), 'w') as f:
        print(cat, file=f)
        print(views, file=f)
        for view in range(views):
            angle = view*360/views
            print(get_name_of_image_file(output_dir, file_id, angle), file=f)
    

            
def find_category(path_to_file, categories):
    splited = path_to_file.split('/')
    for directory in splited:
        if directory in categories: 
            return categories[directory]             
            

def files_to_images(files, id, views, output_dir, categories):
    print("STARTING {}".format(id))
    for file in files:
        render_model(file,id, views, output_dir, categories)
    print("ENDING {}".format(id))

def save_for_mvcnn(args, files, categories):
    size = len(files) // args.t
    pool = []
    if len(files) > 20:
        for i in range(args.t-1):
            p = Process(target=files_to_images, args=(files[i*size:(i+1)*size], i, args.v, args.o, categories))
            p.start()
            pool.append(p)
        p = Process(target=files_to_images,  args=(files[(args.t-1)*size:], args.t-1, args.v, args.o, categories))
        p.start()
        pool.append(p)
        for p in pool:
            p.join()
    else:
        files_to_images(files[i*size:(i+1)*size], 0, args.v, args.o, categories)

def collect_files(files, split, args):
    print("COLLECTING")
    datasets = ['train', 'test', 'val']
    for dataset in range(len(datasets)):
        with open ('{}/{}.txt'.format(args.o, datasets[dataset]), 'w') as f:
            print('{}/{}.txt'.format(args.o,dataset))
            for file in files:
                file_id = file.split('/')[-3]
                if file_id in split and split[file_id] == dataset:
                    print(get_name_of_txt_file(args.o, file_id), file = f)
            
                 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=12, type=int, help="Number of views to render")
    parser.add_argument("-d", type=str, help="root directory of files to be rendered")
    parser.add_argument("-t", default = 8, type=int, help="Number of threads")
    parser.add_argument("-o", type=str, default=".", help="directory of the output files")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")
    parser.add_argument("--dataset",default ="shapenet", type=str, help="Dataset to convert:shapenet or modelnet")
    
    args = parser.parse_args()
    if args.dataset == "shapenet":
        files = find_files(args.d, 'obj')
        categories, split = get_shapenet_metadata(args.d)
    elif args.dataset == "modelnet":
        files = find_files(args.d, 'off')
        categories, split = get_modelnet_metadata(args.d, files)
    
    save_for_mvcnn(args,files, categories)
    collect_files(files, split, args)
    
    