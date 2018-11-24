from __future__ import print_function
import numpy as np
from obj_files import find_files
from obj_files import read_obj_file


def traingle_area(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
def normalize(vector):
    return vector / np.sum(vector)

def mesh_to_point_cloud(points, triangles, n):
    
    distribution = find_area_distribution(points, triangles)
    chosen = np.random.choice(range(len(triangles)),size=n, p=distribution)
    chosen = points[triangles[chosen]]

    u = np.random.rand(n,1)
    v = np.random.rand(n,1)   
    is_outside = u + v > 1
    u[is_outside] = 1 - u[is_outside]
    v[is_outside] = 1 - v[is_outside]
    w = 1 - (u+v)

    xs = chosen[:, 0 ,:] * u
    ys = chosen [:, 1, :] *v
    zs = chosen[:, 2, :] * w

    return (xs + ys +zs)

def find_area_distribution(points, triangles):   
    distribution = np.zeros((len(triangles)))
    for t in range(len(triangles)):
        triangle = triangles[t]
        v1,v2,v3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        distribution[t] = traingle_area(v1, v2, v3)
    return normalize(distribution)

def obj_to_pointcloud(file, args):
    points, triangles, quads = read_obj_file(file)
    return mesh_to_point_cloud(points, triangles, args.n)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=2048, type=int, help="Number of points to smaple")
    parser.add_argument("-d", type=str, help="root directory of .obj files to be voxelizes")
    parser.add_argument("-t", default = 10, type=int, help="Number of threads")
    parser.add_argument("-o", type=str, help="directory of the output files")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")
    args = parser.parse_args()
    files = find_files(args, 'obj')
    for file in files:
        points, triangles, quads = read_obj_file(file)
        pointcloud = mesh_to_point_cloud(points, triangles, n)
        

    
    
    