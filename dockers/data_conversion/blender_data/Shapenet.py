from __future__ import print_function
import os
import json
import numpy as np
from mesh_files import find_files

decoding ={
    0:'train',
    1:'test'
    }

def parse_shapenet_split(root_dir, categories):
    files = find_files(root_dir,'obj')
    split = {}
    coding = {
        'train':0,
        'test':1,
        }
    counter = [0]*55

    for file in files:
        id = get_file_id(file)
        cat = categories[id]
        if counter[cat]%10 in [0,5]:
            split[id] = 1
        else:
            split[id]=0
        counter[cat]+=1
    return split


def get_shapenet_labels(root_dir):
    categories = {}    
    dirs = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,f))])
    for i, dir in enumerate(dirs):
        directory = os.path.join(root_dir, dir)
        ids = os.listdir(directory)
        for id in ids:
            if id not in categories:
                categories[id] = i
    return categories
        

def get_cat_names(root_dir, return_dirnames=False):
    jsonfile = os.path.join(root_dir, 'taxonomy.json')
    cat_names = []
    dirs = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,f))])
    with open(jsonfile) as f:
        data = json.load(f)
        for dir in dirs:
            for dato in data:
                if dato['synsetId'] == dir:
                    cat_names.append(dato['name'].split(',')[0])
    cat_names = [name.replace(' ', '_') for name in cat_names]
    if not return_dirnames:
        return cat_names
    else:
        return cat_names, dirs


def write_cat_names(root_dir, dest):
    with open(os.path.join(dest, "cat_names.txt"), 'w') as f:
        categories = get_cat_names(root_dir)
        for cat in categories:
            print(cat, file=f)
            

def get_metadata(shapenet_root_dir):
    categories = get_shapenet_labels(shapenet_root_dir)
    split = parse_shapenet_split(shapenet_root_dir, categories)
    cat_names = get_cat_names(shapenet_root_dir)
    return categories, split, cat_names

def get_file_id(file):
    return file.split('/')[-3]

def get_files_list(root_dir, categories):
    files = find_files(root_dir, 'obj')
    _, dirnames = get_cat_names(root_dir, return_dirnames=True)
    newfiles = []
    for file in files:
        id = get_file_id(file)
        cat = categories[id]
        if file.split('/')[-4] == dirnames[cat]:
            newfiles.append(file)
    return newfiles
    
if __name__ == "__main__":
    categories, split, _ = get_metadata('/dataset')
    files = get_files_list('/dataset', categories)

    cat_names, dirnames = get_cat_names('/dataset', return_dirnames=True)
    total = 0
    a = {0:0, 1:0,2:0}
    for key in split.keys():
        a[split[key]] +=1
        total+=1
    cat_count_train = {}
    cat_count_test = {}
    cat_counts = [cat_count_train, cat_count_test]
    for cat in range(len(cat_names)):
        for j in [0,1]:
            cat_counts[j][cat] = 0
        
    for file in files:
        id = get_file_id(file)
        cat_counts[split[id]][categories[id]]+=1 
        
    with open('/dataset/shapenetsummary.csv', 'w') as f:
        print('Category & Train & Test \\\\', file=f)
        for i,cat in enumerate(cat_names):
            trct = cat_counts[0][i]
            tect = cat_counts[1][i]
            print("{} & {} & {} \\\\".format(cat.replace('_',' '),trct,tect), file=f)
        print('Total & {} & {} \\\\'.format(len(files)-a[1], a[1]), file=f)
    
    with open('/dataset/shapenetsplit.csv', 'w') as f:
        print('ID,Dataset,Category,Category ID', file=f)
        for file in files:
            id = get_file_id(file)
            cat = categories[id]
            print('{},{},{},{}'.format(id,decoding[split[id]], cat, cat_names[cat]), file=f)
            
        
    
