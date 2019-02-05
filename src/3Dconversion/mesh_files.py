from __future__ import print_function
import os
import numpy as np
from pathlib import Path

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
    return sorted(ext_files)

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
    vertices = rescale_to_unit_sphere(centralize(np.array(vertices)))
    return vertices, np.array(triangles), np.array(quads)

def read_off_file(filename):
    vertices = []
    triangles = []
    quads = []
    with open(filename, 'r') as f:
        line = f.readline().strip()
        if line=='OFF':
           line = f.readline() 
        else:
            line = line[3:]

        n_vertices, n_faces, _ = [int(x) for x in line.split()]
        for _ in range(n_vertices):
            line = f.readline()
            vertices.append([float(x) for x in line.split()])
        for _ in range(n_faces):
            line = f.readline()
            splited = line.split()
            if splited[0] == "3":
                triangles.append([int(x) for x in splited[1:4]])
            elif splited[0] == "4":
                quads.append([int(x) for x in splited[1:5]])
    vertices = rescale_to_unit_sphere(centralize(np.array(vertices)))
    return vertices, np.array(triangles), np.array(quads)
    

def rescale_to_unit_sphere(vertices):
    return vertices / np.max(np.linalg.norm(vertices, axis=1))

def centralize(vertices):

    maxx = np.max(vertices, axis=0)
    minn = np.min(vertices, axis=0)
    centroid = (maxx + minn) / 2
    return vertices - centroid

def off2obj(file):
    vertices, triangles, quads = read_off_file(file)
    obj_file_name = os.path.join(os.path.split(file)[0] , Path(file).stem + ".obj")
    with open(obj_file_name, 'w') as f:
        
        for xyz in vertices:
            f.write('v {:6f} {:6f} {:6f}\n'.format(xyz[0],xyz[2],xyz[1]))
        f.write('\n')
        for ijk in triangles:
            f.write('f %d %d %d\n' % (ijk[0]+1, ijk[1]+1, ijk[2]+1))
        for ijkl in quads:
            f.write('f %d %d %d %d\n' % (ijkl[0]+1, ijkl[1]+1, ijkl[2]+1, ijkl[3]+1))
        
    