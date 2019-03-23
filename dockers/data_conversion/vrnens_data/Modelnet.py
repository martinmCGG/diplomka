from __future__ import print_function
from pathlib import Path
from mesh_files import find_files
import os

decoding = {
    0:'train',
    1:'test'
    }

def get_metadata(root_dir):
    files = find_files(root_dir, 'off')
    categories = {}
    split = {}
    coding = {
        'train':0,
        'test':1
        }
    cats = get_cat_names(root_dir)
    cats_mapping = {}
    for i in range(len(cats)):
        cats_mapping[cats[i]] = i
    for file in files:
        splited = file.split('/')
        name = os.path.join(Path(file).stem)
        categories[name] = cats_mapping[splited[-3]]
        split[name] = coding[splited[-2]]
    cat_names = get_cat_names(root_dir)
    return categories, split, cat_names

def get_cat_names(folder):
    cats = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder,x))]
    if "converted" in cats:
        cats.remove("converted")
        print("converted not included into categories")
    return sorted(cats)

def write_cat_names(root_dir, dest):
    print("writing")
    with open(os.path.join(dest, "cat_names.txt"), 'w') as f:
        categories = get_cat_names(root_dir)
        for cat in categories:
            print(cat, file=f)

def get_files_list(root_dir, categories):
    return find_files(root_dir, 'off')

def get_file_id(file):
    return file.split('/')[-1].split('.')[-2]

if __name__ == "__main__":
    categories, split, cat_names = get_metadata('/dataset')
    files = get_files_list('/dataset', categories)
    print('ALL files ',len(files))
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
        
    with open('/dataset/modelnetsummary.csv', 'w') as f:
        print('Category & Train & Test \\\\', file=f)
        for i,cat in enumerate(cat_names):
            trct = cat_counts[0][i]
            tect = cat_counts[1][i]
            print("{} & {} & {} \\\\".format(cat.replace('_',' '),trct,tect), file=f)
        print('Total & {} & {} \\\\'.format(len(files)-a[1], a[1]), file=f)
    
    with open('/dataset/modelnetsplit.csv', 'w') as f:
        print('ID,Dataset,Category,Category ID', file=f)
        for file in files:
            id = get_file_id(file)
            cat = categories[id]
            print('{},{},{},{}'.format(id,decoding[split[id]], cat, cat_names[cat]), file=f)
            