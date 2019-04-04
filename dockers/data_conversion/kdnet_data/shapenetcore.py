from __future__ import print_function
import os
import numpy as np
import h5py as h5
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
    
       
def save_for_kdnet_multi(files,config, categories, split):
    procesor = MultiProcesor(files, config.num_threads, config.log_file, categories, split, config.dataset_type, load_mesh, write_for_kdnet)
    procesor.run(config._asdict())

def save_for_kdnet(files, config, categories, split):
    path2data = config.data
    path2save = config.output
    
    train_filenames =  [file for file in files if get_file_id(file) not in split or split[get_file_id(file)] == 0]
    test_filenames = [file for file in files if get_file_id(file) in split and split[get_file_id(file)] == 1]

    train_vertices_cnt = 0
    train_faces_cnt = 0

    test_vertices_cnt = 0
    test_faces_cnt = 0

    for i, shapefile in enumerate(train_filenames):
        with open(shapefile, 'r') as fobj:
            for line in fobj:
                if 'v' in line:
                    train_vertices_cnt += 1
                if 'f' in line:
                    train_faces_cnt += 1


    for i, shapefile in enumerate(test_filenames):
        with open(shapefile, 'r') as fobj:
            for line in fobj:
                if 'v' in line:
                    test_vertices_cnt += 1
                if 'f' in line:
                    test_faces_cnt += 1


    train_nFaces = np.zeros((1 + len(train_filenames),), dtype=np.int32)
    train_faces = np.zeros((train_faces_cnt, 3), dtype=np.int32)
    train_vertices = np.zeros((train_vertices_cnt, 3), dtype=np.int32)
    train_labels = np.zeros((len(train_filenames),), dtype=np.int8)

    test_nFaces = np.zeros((1 + len(test_filenames),), dtype=np.int32)
    test_faces = np.zeros((test_faces_cnt, 3), dtype=np.int32)
    test_vertices = np.zeros((test_vertices_cnt, 3), dtype=np.int32)
    test_labels = np.zeros((len(test_filenames),), dtype=np.int8)
    data = [train_nFaces, train_faces, train_vertices, train_labels, test_nFaces, test_faces, test_vertices, test_labels]
    
    train_nFaces[0] = 0
    test_nFaces[0] = 0

    vertices_pos = 0
    faces_pos = 0
    for i, shapefile in enumerate(train_filenames):
        print(shapefile)
        shape_name = get_file_id(shapefile)
        
        shape_vertices, shape_faces, _ = read_obj_file(shapefile)
        
        buf = shape_vertices[:, 1].copy()
        shape_vertices[:, 1] = shape_vertices[:, 2]
        shape_vertices[:, 2] = buf
        shape_faces = np.array(shape_faces) - 1

        vertices_offset = shape_vertices.shape[0]
        faces_offset = shape_faces.shape[0]

        train_vertices[vertices_pos:vertices_pos+vertices_offset] = shape_vertices
        train_faces[faces_pos:faces_pos+faces_offset] = vertices_pos + shape_faces
        train_nFaces[i+1] = faces_pos + faces_offset
        train_labels[i] = categories[shape_name]

        vertices_pos += vertices_offset
        faces_pos += faces_offset

    vertices_pos = 0
    faces_pos = 0
    
    for i, shapefile in enumerate(test_filenames):
        print(shapefile)
        shape_name = get_file_id(shapefile)
        shape_vertices, shape_faces, _ = read_obj_file(shapefile)

        shape_vertices = np.array(shape_vertices)
        buf = shape_vertices[:, 1].copy()
        shape_vertices[:, 1] = shape_vertices[:, 2]
        shape_vertices[:, 2] = buf
        shape_faces = np.array(shape_faces) - 1
        
        vertices_offset = shape_vertices.shape[0]
        faces_offset = shape_faces.shape[0]

        test_vertices[vertices_pos:vertices_pos+vertices_offset] = shape_vertices
        test_faces[faces_pos:faces_pos+faces_offset] = vertices_pos + shape_faces
        test_nFaces[i+1] = faces_pos + faces_offset
        test_labels[i] = categories[shape_name]
        
        vertices_pos += vertices_offset
        faces_pos += faces_offset

    path2save = config.output
    names = ['train_nFaces','train_faces','train_vertices','train_labels','test_nFaces','test_faces','test_vertices','test_labels']

    hf = h5.File(path2save + '/data' + '.h5', 'w')
    for i, name in enumerate(names):
        print(name)
        hf.create_dataset(name, data=data[i])
    hf.close()
    print('ENDING')

def prepare(config):
    with open(config.log_file, 'w') as f:
        print('Starting', file = f)

    categories, split, cat_names = get_metadata(config.data)
    files = get_files_list(config.data, categories)
    write_cat_names(config.data, config.output)
    save_for_kdnet_multi(files, config, categories, split)
    merge_h5_files(config.output)
    
    filename = os.path.join(config.output,'data.h5')
    hf = h5.File(filename,'r')
    print(list(hf.keys()))
    hf.close()
        