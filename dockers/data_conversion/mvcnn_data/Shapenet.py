from __future__ import print_function
import os
import json
import numpy as np
from mesh_files import find_files


def parse_shapenet_split(csv_file):
    split = {}
    coding = {
        'train':0,
        'test':1,
        'val':2
        }
    with open(csv_file, 'r') as f:
        f.readline()
        for line in f:
            splited = line.strip().split(',')
            split[splited[-2]] = coding[splited[-1]]
    return split


def get_shapenet_labels(root_dir):
    categories = {}
    dirs = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,f))])
    for i in range(len(dirs)):
        directory = os.path.join(root_dir, dirs[i])
        ids = os.listdir(directory)
        for id in ids:
            categories[id] = i
    return categories
        

def get_cat_names(root_dir):
    jsonfile = os.path.join(root_dir, 'taxonomy.json')
    cat_names = []
    dirs = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,f))])
    with open(jsonfile) as f:
        data = json.load(f)
        for dir in dirs:
            for dato in data:
                if dato['synsetId'] == dir:
                    cat_names.append(dato['name'].split(',')[0])
    return cat_names

def write_cat_names(root_dir, dest):
    with open(os.path.join(dest, "cat_names.txt"), 'w') as f:
        categories = get_cat_names(root_dir)
        for cat in categories:
            print(cat, file=f)
            

def get_metadata(shapenet_root_dir):
    splitfile = os.path.join(shapenet_root_dir, 'all.csv')
    split = parse_shapenet_split(splitfile)
    categories = get_shapenet_labels(shapenet_root_dir)
    cat_names = get_cat_names(shapenet_root_dir)
    return categories, split


