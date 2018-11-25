'''
Created on 25. 11. 2018

@author: miros
'''
from __future__ import print_function
import re
import os
import numpy as np

def find_re_files(directory, pattern, output):
    dict = {}
    files = [file for file in os.listdir(directory) if re.match(pattern, file)]
    print ("Found {} files for regex {}".format(len(files), pattern))
    for file in files:
        arch = np.load(os.path.join(directory, file))
        for key in arch.keys():
            if key not in dict:
                dict[key] = []
            dict[key].append(arch[key])
            #print(arch[key].shape)
    
    bigdict = {}
    for key in dict.keys():
        arr = np.array(dict[key])
        shape = list(arr.shape)
        shape = shape[2:]
        shape = [-1] + shape
        arr = np.reshape(arr, shape)
        bigdict[key] = arr
    np.savez(output, **bigdict)
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="root directory of npz files to be joined")
    parser.add_argument("-re",  help="Regular expression")
    parser.add_argument("-o", type=str, help="name of the output file")

    args = parser.parse_args()
    find_re_files(args.d, args.re, args.o)