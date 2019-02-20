import bpy
import os
import math
import numpy as np
import glob

from os import listdir
from os.path import isfile, join

scene = bpy.context.scene
context = bpy.context

def get_name_of_image_file(output_dir, file_id, i):
    return os.path.join(output_dir , file_id, file_id +'_'+ str(i) + ".png")

def delete_cube():
    #Delete default cube
    objs = bpy.data.objects
    objs.remove(objs["Cube"], True)


def render_one_model(model_path, file_id, output_dir, nviews=12, resolution=224):
    
    bpy.ops.import_scene.obj(filepath=model_path, filter_glob="*.obj")

    #Set resolution
    scene = bpy.data.scenes["Scene"]
    # Set render resolution
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100 
    
    imported = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    maxDimension = 5.0
    scaleFactor = maxDimension / max(imported.dimensions)
    imported.scale = (scaleFactor,scaleFactor,scaleFactor)
    imported.location = (0, 0, 0)
    
    imported.rotation_mode = 'XYZ'
    imported.rotation_euler[1] = np.pi
    views = np.linspace(0, 2*np.pi, nviews, endpoint=False)

    
    for i in range(nviews):
        imported.rotation_euler[2] = views[i]
        #imported.rotation_euler[0] = np.pi
        filename = model_path.split("/")[-1]
        print (filename)
        bpy.ops.view3d.camera_to_view_selected()
        context.scene.render.filepath = get_name_of_image_file(output_dir, file_id, i)
        bpy.ops.render.render( write_still=True )
        
    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)
    bpy.ops.object.delete()
    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)
    
    imported = None
    del imported       
