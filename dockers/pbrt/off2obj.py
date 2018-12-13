#Convert files from off to obj format
from off_files import read_off_file
from pathlib import Path
import os

def off2obj(file):
    vertices, triangles, quads = read_off_file(file)
    obj_file_name = os.path.join(os.path.split(file)[0] , Path(file).stem + ".obj")
    print(obj_file_name)
    with open(obj_file_name, 'w') as f:
        for xyz in vertices:
            f.write('v %g %g %g\n' % tuple(xyz))
        f.write('\n')
        for ijk in triangles:
            f.write('f %d %d %d\n' % (ijk[0]+1, ijk[1]+1, ijk[2]+1))
        for ijkl in quads:
            f.write('f %d %d %d %d\n' % (ijkl[0]+1, ijkl[1]+1, ijkl[2]+1, ijkl[3]+1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, default=".", help="obj file to be converted")
    args = parser.parse_args()
    off2obj(args.file)