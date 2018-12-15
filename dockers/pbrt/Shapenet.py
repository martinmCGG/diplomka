from __future__ import print_function
import os
import json
import numpy as np

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

def get_shapenet_labels(jsonfile):
    categories = {}
    with open(jsonfile) as f:
        data = json.load(f)
    for dato in data:
        id = dato['synsetId']
        if id not in categories:
            category = dato['name'].split(',')[0]
            categories[id] = category
            _set_labels_for_children(categories, dato, data, category)
    
    unique_categories =sorted(list(set(categories.values())))
    categories_mapping = {}
    for i in range(len(unique_categories)):
        categories_mapping[unique_categories[i]] = i 
    for key in categories.keys():
        categories[key] = categories_mapping[categories[key]]
    return categories

def _set_labels_for_children(categories, parent, data, category):
    children = parent['children']
    for child in children:
        for dato in data:
            if dato['synsetId'] == child:
                categories[child] = category
                _set_labels_for_children(categories, dato, data, category)
                
def get_shapenet_metadata(shapenet_root_dir):
    jsonfile = os.path.join(shapenet_root_dir, 'taxonomy.json')
    splitfile = os.path.join(shapenet_root_dir, 'all.csv')
    categories = get_shapenet_labels(jsonfile)
    split = parse_shapenet_split(splitfile)
    return categories, split



    

