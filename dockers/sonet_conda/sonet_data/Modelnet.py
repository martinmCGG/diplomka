from __future__ import print_function
from pathlib import Path
import os

def get_metadata(root_dir, files):
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
    return categories, split

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