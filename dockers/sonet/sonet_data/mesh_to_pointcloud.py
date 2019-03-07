from __future__ import print_function
import numpy as np
from mesh_files import *

def triangle_normal(v1,v2,v3):
    return np.cross(v2-v1, v3-v1)

def traingle_area(v1, v2, v3):
    return 0.5 * np.linalg.norm(triangle_normal(v1, v2, v3))

def normalize(vector):
    return vector / np.sum(vector)

def mesh_to_point_cloud(points, triangles, n, normal=True):
    distribution = find_area_distribution(points, triangles)
    chosen = np.random.choice(range(len(triangles)),size=n, p=distribution)
    chosen_points = points[triangles[chosen]]
    normals = [triangle_normal(*triangle) for triangle in chosen_points] if normal else None
    u = np.random.rand(n,1)
    v = np.random.rand(n,1)
    is_outside = u + v > 1
    u[is_outside] = 1 - u[is_outside]
    v[is_outside] = 1 - v[is_outside]
    w = 1 - (u+v)
    xs = chosen_points[:, 0 ,:] * u
    ys = chosen_points [:, 1, :] *v
    zs = chosen_points[:, 2, :] * w
    
    if normals: 
        return np.concatenate(((xs+ys+zs),normals), axis=1)
    else:
        return np.array(xs + ys + zs)


def find_area_distribution(points, triangles):   
    distribution = np.zeros((len(triangles)))
    for t in range(len(triangles)):
        triangle = triangles[t]
        v1,v2,v3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        distribution[t] = traingle_area(v1, v2, v3)
    return normalize(distribution)

def file_to_pointcloud(filename, type, args):
    if type == 'shapenet':
        points, triangles, quads = read_obj_file(filename)
    elif type == 'modelnet':
        points, triangles, quads = read_off_file(filename)

    return mesh_to_point_cloud(points, triangles, args.n, normal=True)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=2048, type=int, help="Number of points to smaple")
    parser.add_argument("-d", type=str, help="root directory of .obj files to be voxelizes")
    parser.add_argument("-t", default = 1, type=int, help="Number of threads")
    parser.add_argument("-o", type=str, help="directory of the output files")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")

    
    args = parser.parse_args()
    files = find_files(args.d, 'obj')
    for file in files:
        points, triangles, quads = read_obj_file(file)
        pointcloud = mesh_to_point_cloud(points, triangles, args.n, normal=True)

        

    
    
    