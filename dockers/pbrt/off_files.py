import numpy as np
from obj_files import find_files

def read_off_file(filename):
    vertices = []
    triangles = []
    quads = []
    with open(filename, 'r') as f:
        line = f.readline()
        line = f.readline()
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
    return rescale_to_unit_sphere(np.array(vertices)), np.array(triangles), np.array(quads)

def rescale_to_unit_sphere(vertices):
    vertices = vertices / np.max(np.abs(vertices))
    return vertices