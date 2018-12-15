from __future__ import print_function
from obj_files import find_files
from obj_files import read_obj_file
import numpy as np

def find_biggest_coord(vertices):
    return np.max(np.abs(vertices))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="root directory of .obj files to be voxelized")
    parser.add_argument("-m", default = 10000, type=int, help="Max number of models to save to one file")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")
        
    args = parser.parse_args()
    files = find_files(args.d, 'obj')
    counter = 0
    with open(args.l, 'w') as log:
        for file in files:
            vertices,_,_ = read_obj_file(file)
            biggest=find_biggest_coord(vertices)
            print(biggest)
            if biggest > 0.7:
                counter+=1
                print(biggest, file=log)
    print("Found {} potential problems".format(counter))
    
    