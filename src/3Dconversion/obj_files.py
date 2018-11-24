from __future__ import print_function
import os
import numpy as np

def find_files(directory,extension):
    print("Scanning for files...")
    fname = os.path.join(directory,"all_{}_files.txt".format(extension))
    ext_files = []
    if not os.path.isfile(fname):
        for folder, subs, files in os.walk(directory):
            for filename in files:
                if filename.split('.')[-1] == extension:
                    filename = os.path.join(folder, filename)
                    ext_files.append(filename)
        with open(fname, 'w') as f:
            for file in ext_files:
                print(file, file=f)
    else:
        with open(fname, 'r') as f:
            for line in f:
                ext_files.append(line.strip())
    
    print("Found {} files".format(len(ext_files)))
    return ext_files

def read_obj_file(filename):
    vertices = []
    triangles = []
    quads = []
    with open(filename, 'r') as f:
        for line in f:
            splited = line.split()
            if splited and splited[0] == 'f':
                size = len(splited) - 1
                point = [] 
                for i in range(size):
                    point.append(int(splited[i+1].split('/')[0])-1)
                if size == 3:
                    triangles.append(point)
                elif size==4:
                    quads.append(point)
            
            elif splited and splited[0] == 'v':
                vertices.append([float(splited[1]), float(splited[3]),float(splited[2])])
    return np.array(vertices), np.array(triangles), np.array(quads)