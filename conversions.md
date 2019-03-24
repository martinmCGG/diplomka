# Data conversion

This page describes the process of converting mesh files to specific formats in more detail. There are three main types of input used for processing 3D models by neural networks - voxels, point clouds and 2D images taken from multiple views. Although the conversion is implemented in multi-threaded fashion it still can be quite time consuming on big datasets, it can take up to ten or more hours in some cases.

## Voxels

**Docker****:** _dockers/data_conversion/vrnens_data_  
**Networks:** _dockers/vrnens_

**Details:** Voxels are natural extension of pixels to three dimensions. Instead of square lattice we have cube occupancy grid. For voxelization of meshes we use [openvdb library](https://www.openvdb.org/), which is written in c++ but offers some basic functionality including voxelization in python. Only one of our networks uses directly voxels as their input and for the purposes of this network data is stored in _.npz_ format. Parameters include `num_voxels` which is resolution of the voxel grid and `num_rotations` which is number of rotations of single 3D model to be voxelized.

## Images

Multi view neural networks get multiple rendered images of a single 3D model as their input. This rendering is computationally most demanding of all types of data conversion. Fortunately all the multi view networks accept the same format.

**Docker****:** _dockers/data_conversion/pbrt_data_  
**Networks:** _dockers/mvcnn,_ _dockers/mvcnn2,_ _dockers/vgg,_ _dockers/rotnet_

This is our implementation of rendering using [pbrt](https://www.pbrt.org/). It can be very slow, although we are sure that the implementation could be improved. You can set the number of rendered views as parameter `num_views`. If the `dodecahedron` parameter is set to True, twenty views from the vertices of regular dodecahedron will be rendered. This is useful when working with dataset without canonical rotation. You can also use more rotations of the camera from a single viewpoint by setting `camera_rotations` parameter. Pbrt can use only _.obj_ files so for ModelNet ._off_ files are converted to this format and saved. Set _remove_obj = False_ if you want to keep these files after rendering is done.

**Docker****:** _dockers/data_conversion/blender_data_  
**Networks:** _dockers/mvcnn,_ _dockers/mvcnn2,_ _dockers/vgg,_ _dockers/rotnet_

With better success we used scripts for blender provided by team around [this paper](https://people.cs.umass.edu/%7Ejcsu/papers/shape_recog/). We just connected these into our framework, the parameters are same as above. There are two options shaded images and depth images. For details consult the original paper. You can set the mode of rendering by setting variable `render` in _run.sh_.

## Point clouds

Point cloud is simply set of points in three dimensional space. It can be obtained fairly easily from meshes by sampling random points from the faces. The probability of the face being selected is weighted by its area.

**Docker****:** _dockers/data_conversion/pnet_data_  
**Networks:** _dockers/pointnet,__dockers/pointnet2_

For pointnet and its successor pointnet++ we save the sampled point data in several _.h5_ files which are listed in text files. With all parameters unchanged, all paths will be valid, but make sure that the paths in these text files are correct inside the docker container. You can set up number of sampled points as a parameter. You can also set `normal = True` to sample the points with surface normals, but the pointnet implementation of this is currently not working.

**Docker****:** _dockers/data_conversion/sonet_data_  
**Networks:** _dockers/sonet_

Sonet data is sampled in the same way as pointnet with surface normals. Then it creates and learns self organizing map to better represent the data. This map is created during the data conversion and therefore this script requires GPU and nvidia runtime. You can set the number of som nodes as parameter `num_nodes` and this should be a square integer.

## `<span style="font-family: serif;"><span style="font-family: monospace;"></span></span>`<span style="font-family: serif;"><span style="font-family: monospace;"></span></span><span style="font-family: serif;"><span style="font-family: monospace;"></span></span><span style="font-family: serif;"><span style="font-family: monospace;"></span></span>Other

**Docker****:** _dockers/data_conversion/octree_data_  
**Networks:** _dockers/pointnet,__dockers/octree,_ _dockers/pointnet,__dockers/octree_adaptive_

Octree is a data structure for storing 3D data efficiently. This script which uses tools provided by the authors of original paper. Data is augmented by creating more than one rotation of the 3D model and you can control this number by setting `num_rotations` parameter. If you want to create data for adaptive octree network set `adaptive = True`. There is also an option to correct some badly written _.off_ files in ModelNet40.

**Docker****:** _dockers/data_conversion/kdnet_data_  
**Networks:** _dockers/pointnet,__dockers/kdnet_

This script only loads the data and saves them in a large _.h5_ file. Construction of kd-trees which are input of the kd-network is contructed during training.

## [**<<< BACK**](README.md)