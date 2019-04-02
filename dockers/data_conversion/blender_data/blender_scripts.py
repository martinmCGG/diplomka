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
    return os.path.join(output_dir, file_id +'_'+ str(i) + ".png")

def render_phong(model_path, file_id, output_dir, nviews=12, resolution=224):
    
    def delete_model(name):
        for ob in scene.objects:
            if ob.type == 'MESH' and ob.name.startswith(name):
                ob.select = True
            else:
                ob.select = False
        bpy.ops.object.delete()

    def init_camera():
        cam = D.objects['Camera']
        # select the camera object
        scene.objects.active = cam
        cam.select = True
    
        # set the rendering mode to orthogonal and scale
        C.object.data.type = 'ORTHO'
        C.object.data.ortho_scale = 2.
    
    def load_model(path):
        d = os.path.dirname(path)
        ext = path.split('.')[-1]
    
        name = os.path.basename(path).split('.')[0]
        # handle weird object naming by Blender for stl files
        if ext == 'stl':
            name = name.title().replace('_', ' ')
    
        if name not in D.objects:
            print('loading :' + name)
            if ext == 'stl':
                bpy.ops.import_mesh.stl(filepath=path, directory=d,
                                        filter_glob='*.stl')
            elif ext == 'off':
                bpy.ops.import_mesh.off(filepath=path, filter_glob='*.off')
            elif ext == 'obj':
                bpy.ops.import_scene.obj(filepath=path, filter_glob='*.obj')
            else:
                print('Currently .{} file type is not supported.'.format(ext))
                exit(-1)
        return name
    
    def do_model(path, output_dir, file_id):
        name = load_model(path)
        #center_model(name)
        #normalize_model(name)
        def move_camera(coord):
            def deg2rad(deg):
                return deg * math.pi / 180.
            r = 3.
            theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
            loc_x = r * math.sin(theta) * math.cos(phi)
            loc_y = r * math.sin(theta) * math.sin(phi)
            loc_z = r * math.cos(theta)
            D.objects['Camera'].location = (loc_x, loc_y, loc_z)
        
        for i, c in enumerate(cameras):
            move_camera(c)
            bpy.ops.render.render()
            D.images['Render Result'].save_render(filepath=get_name_of_image_file(output_dir, file_id, i))

        delete_model(name)
        
    def fix_camera_to_origin():
        origin_name = 'Origin'
        # create origin
        try:
            origin = D.objects[origin_name]
        except KeyError:
            bpy.ops.object.empty_add(type='SPHERE')
            D.objects['Empty'].name = origin_name
            origin = D.objects[origin_name]
        origin.location = (0, 0, 0)
    
        cam = D.objects['Camera']
        scene.objects.active = cam
        cam.select = True
    
        if 'Track To' not in cam.constraints:
            bpy.ops.object.constraint_add(type='TRACK_TO')
    
        cam.constraints['Track To'].target = origin
        cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
        cam.constraints['Track To'].up_axis = 'UP_Y'
    
    C = bpy.context
    D = bpy.data
    scene = D.scenes['Scene']
    cameras = [(60, i) for i in range(0, 360, 30)]
    render_setting = scene.render
    # Set render resolution
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100 
    init_camera()
    fix_camera_to_origin()
    do_model(model_path, output_dir, file_id)

def render_one_model(model_path, file_id, output_dir, nviews=12, resolution=224):
    print('rendering')
    bpy.ops.import_scene.obj(filepath=model_path, filter_glob='*.obj')

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
