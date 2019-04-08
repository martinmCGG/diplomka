from __future__ import print_function
import os
import numpy as np
import h5py as h5
import traceback
import re
from mesh_files import *
from Shapenet import *
from MultiProcesor import MultiProcesor
from npz_join import join_h5


def write_for_kdnet(buffer, buffer_cats, dataset, id, config):
    nFaces = [0]
    all_faces = []
    all_vertices = []
    offset = 0
    for vertices, faces in buffer:
        offset+= len(faces)
        nFaces.append(offset)
        all_faces += list(faces)
        all_vertices+=list(vertices)
    names = ['{}_nFaces','{}_faces','{}_vertices','{}_labels']
    data = [nFaces, all_faces, all_vertices, buffer_cats]
    data = [np.array(dato) for dato in data]
    hf = h5.File(os.path.join(config.output, dataset + '_data_' + str(id) +'.h5') , 'w')
    for dato, name in zip(data, names):
        name = name.format(dataset)
        hf.create_dataset(name, data=dato)
    hf.close()
    
    
def load_mesh(filename, type, args):
    shape_vertices, shape_faces, _ = read_obj_file(filename)
    return [shape_vertices, shape_faces]

      
def merge_h5_files(directory):
    join_h5(directory, '.*\.h5', 'data.h5')
    
       
def save_for_kdnetfiles,config, categories, split):
    procesor = MultiProcesor(files, config.num_threads, config.log_file, categories, split, config.dataset_type, load_mesh, write_for_kdnet)
    procesor.run(config._asdict())

def log(file, log_string):
    with open(file, 'a') as f:
        print(log_string)
        print(log_string, file=f)

def prepare(config):
    with open(config.log_file, 'w') as f:
        print('Starting', file = f)
        print('Starting')
    try:
        categories, split, cat_names = get_metadata(config.data)
        files = get_files_list(config.data, categories)
        write_cat_names(config.data, config.output)
    except:
        err_string = traceback.format_exc()
        log(config.log_file, "Exception occured while reading files.")
        log(config.log_file, err_string)   
        sys.exit(1)
    save_for_kdnet(files, config, categories, split)
    log(config.log_file, 'Merging h5 files.')
    merge_h5_files(config.output)
    log(config.log_file, 'FINISHED') 
        