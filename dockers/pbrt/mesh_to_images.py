import os
from pathlib import Path
from obj_files import find_files
    
def render_one_image(obj_file):
    pbrt_file = os.path.join(os.path.split(obj_file)[0] , Path(obj_file).stem + ".pbrt")
    print(pbrt_file)
    cmd = "./obj2pbrt {} {}".format(obj_file, pbrt_file)
    print (cmd)
    os.system(cmd) 




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=10, type=int, help="Number of views to render")
    parser.add_argument("-d", type=str, help="Absolute path to root directory of .obj files to be rendered")
    parser.add_argument("-t", default = 4, type=int, help="Number of threads")
    parser.add_argument("-m", default = 10000, type=int, help="Max number of models to save to one file")
    parser.add_argument("-o", type=str, default=".", help="directory of the output files")
    parser.add_argument("-l",default ="log.txt", type=str, help="logging file")
    parser.add_argument("--dataset",default ="shapenet", type=str, help="Dataset to convert:currently supportedL")
    parser.add_argument("--image",default ="pbrt_w", type=str, help="Name of the docker image")
    
    args = parser.parse_args()
    
    '''if args.dataset == "shapenet":
        files = find_files(args.d, 'obj')
        categories, split = get_shapenet_metadata(args.d)
    elif args.dataset == "modelnet":
        files = find_files(args.d, 'off')
        categories, split = get_modelnet_metadata(args.d, files)'''
    files = find_files("/data", 'obj')
    render_one_image(files[0])
    