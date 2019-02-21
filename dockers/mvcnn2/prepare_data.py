from __future__ import print_function
import os

def copy(args, name):
    d = os.path.join(args.d, 'converted')
    folders = [os.path.join(args.o, cat, name) for cat in cats]
    with open(os.path.join(d, name+'.txt'), 'r') as f:
        for line in f:
            folder = folders[int(line.split()[1].strip())]
            id = line.split('/')[-2]
            with open(line.split()[0], 'r') as f2:
                for line2 in f2:
                    os.system('cp {} {}'.format(line2.strip(),folder))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default='/out', help="root directory of dataset")
    parser.add_argument("-o", type=str, default='/mvcnn2/logs/Modelnet40A_mvcnn2', help="directory of the output files")    
    args = parser.parse_args()
    
    with open(os.path.join(args.d, 'cat_names.txt'), 'r') as f:
        cats = [line.strip() for line in f.readlines()]
    print(cats)
    
    for cat in cats:
        os.system('mkdir -m 777 {}'.format(os.path.join(args.o, cat)))
        for d in ['test', 'train']:
            os.system('mkdir -m 777 {}'.format(os.path.join(args.o, cat, d)))
    
    copy(args, 'train')
    copy(args, 'test')

        