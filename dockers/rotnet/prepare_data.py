import os
import argparse
import random

VIEWS = 12
NCATS = 40

def find_all_files(folder, dataset):
    print('starting')
    list = []
    categories = sorted(os.listdir(folder))
    for cat in categories:
        files = sorted(os.listdir(os.path.join(folder,cat,dataset)))
        for i in range(len(files)//VIEWS):
            one_model = files[i*VIEWS:(i+1)*VIEWS]
            rot_index = random.randint(0,VIEWS)
            if dataset == 'train':
                #one_model = one_model[rot_index:]+one_model[0:rot_index]
                random.shuffle(one_model)
            list.append(one_model)
            one_model.append(cat)
            one_model.append(categories.index(cat))
    return list

def prepare_data(file):
    all_files = []
    with open(file, "r") as f:
        for line in f:
            line2 = line.split()[0]
            with open(line2) as f2:
                cat = f2.readline()
                f2.readline().strip()
                for view in range(VIEWS):
                    to_append = "{} {}".format(f2.readline().strip(),cat)
                    all_files.append(to_append)
                    
    with open(file.split('.')[0]+"rotnet.txt", 'w') as f:
        for line in all_files:  
            f.write(line)
        
        
    
    
if __name__ == "__main__":
    """import sys
    dataset = sys.argv[1]
    folder = sys.argv[2]
    print(dataset, folder)
    all_files = find_all_files(folder, dataset)
    if dataset == 'train':
        random.shuffle(all_files)
    with open('{}_shuffled'.format(dataset), 'w') as f:
        for file in all_files:
            filed = file[:-3]
            catindex = file[-1]
            cat = file[-2]
            for image in filed:
                print(os.path.join(folder,cat,dataset,image) + " " + str(catindex), file=f)"""
    
    import sys
    prepare_data(sys.argv[1])
                
    
    