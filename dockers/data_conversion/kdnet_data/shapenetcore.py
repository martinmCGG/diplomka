from __future__ import print_function
import os
import numpy as np
import pandas as pd
import h5py as h5
import Shapenet
import re
from mesh_files import *

def get_file_id(file):
    return file.split('/')[-3]


def  save_for_kdnet(files, config, categories, split):

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
    data = [train_nFaces,train_faces,train_vertices,train_labels,test_nFaces,test_faces,test_vertices,test_labels]
    
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
    print('Starting')
    path2data = config.data
    path2save = config.output
    categories, split = Shapenet.get_metadata(path2data)
    files = find_files(path2data, 'obj')
    categories, split = Shapenet.get_metadata(path2data)
    Shapenet.write_cat_names(path2data, path2save)
    save_for_kdnet(files,config, categories, split)


        