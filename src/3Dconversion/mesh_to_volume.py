from __future__ import print_function
import numpy as np
import os
from obj_files import read_obj_file
import pyopenvdb as vdb


def convert_to_grid(vertices, triangles, quads, grid_size):
    if len(triangles) == 0:
        triangles = None
    if len(quads) == 0:
        quads = None
    transform = vdb.createLinearTransform(voxelSize=(1.0/(grid_size-2)))
    grid = vdb.FloatGrid.createLevelSetFromPolygons(vertices, triangles = triangles, quads = quads, transform=transform, halfWidth = 1)
    
    outside = grid.background
    width = 2.0 * outside 
    # Visit and update all of the grid's active values, which correspond to
    # voxels in the narrow band.
    for iter in grid.iterOnValues():
        dist = iter.value
        iter.value = (outside - dist) / width
    # Visit all of the grid's inactive tile and voxel values and update
    # the values that correspond to the interior region.
    for iter in grid.iterOffValues():
        if iter.value < 0.0:
            iter.value = 1.0

    grid.background = 0.0
    return grid


def voxels_to_nparray(grid, grid_size):
    accessor = grid.getAccessor()
    arr = np.zeros((grid_size,grid_size,grid_size), dtype=np.uint8) 
    start = -grid_size // 2 + 1
    stop = grid_size//2 + 1
    for i in range(start, stop):
        for j in range(start, stop):
            for k in range(start, stop):
                ijk = (i,j,k)
                value = accessor.probeValue(ijk)[0]
                arr[i-start,j-start,k-start] = 1 if value > 0 else 0
    return arr

def obj_to_voxel_array(filename, args):
    array = []
    vertices, triangles, quads = read_obj_file(filename)
    for _ in range(args.r):
        vertices = vertices.dot(args.matrix)
        grid = convert_to_grid(vertices, triangles, quads, args.v)
        array.append(voxels_to_nparray(grid, args.v))
    return array


