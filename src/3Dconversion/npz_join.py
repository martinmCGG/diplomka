from __future__ import print_function
import re
import os
import numpy as np
import h5py
#python npz_join.py -d /home/krabec/Data/ShapeNetPoints/ -o /home/krabec/Data/ShapeNet/data.h5 -f h5 -re ".*val.*.h5"

def find_re_files(directory, pattern):
    files = [os.path.join(directory,file) for file in os.listdir(directory) if re.match(pattern, file)]
    print ("Found {} files for regex {}".format(len(files), pattern))
    return files

def join_npz(files, output):
    dict = {}
    for file in files:
        arch = np.load(file)
        for key in arch.keys():
            if key not in dict:
                dict[key] = []
            dict[key].append(arch[key])
            print(arch[key].shape)
    bigdict = {}
    for key in dict.keys():
        arr = np.concatenate(dict[key])
        bigdict[key] = arr
    print("Saving...")
    np.savez(output, **bigdict)
    print("Saved...")
    

def join_h5(files, output):
    dict = {}
    for file in files:
        arch = h5py.File(file, 'r')
        for key in arch.keys():
            if key not in dict:
                dict[key] = []
            dataset = arch.get(key)
            copy = np.copy(np.array(dataset))
            dict[key].append(copy)
    hf = h5py.File(output, 'w')
    for key in dict.keys():
        arr = np.concatenate(dict[key])
        spec_dtype = h5py.special_dtype(vlen=np.dtype('float64'))
        hf.create_dataset(key, data=arr, dtype=spec_dtype)
    hf.close()
        
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="root directory of npz files to be joined")
    parser.add_argument("-re",  help="Regular expression")
    parser.add_argument("-o", type=str, help="name of the output file")
    parser.add_argument("-f", type=str,default='npz', help="npz or h5 format")

    args = parser.parse_args()
    files = find_re_files(args.d, args.re)
    if args.f == 'npz':
        join_npz(files, args.o)
    elif args.f == 'h5':
        join_h5(files, args.o)