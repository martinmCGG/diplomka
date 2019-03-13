import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default= "D:\\workspace\\diplomka\\src\\3Dconversion\\test_0_0.h5", type=str, help="npz file to load")
    args = parser.parse_args()
    
    f = h5py.File(args.f)
    print(f)
    index = 0
    labels = np.array(f['label'])
    points = np.array(f['data'])
    soms = np.array(f['som'])
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.scatter(points[index,:,0], points[index,:,1],points[index,:,2])
    ax.scatter(soms[index,:,0], soms[index,:,1],soms[index,:,2])
    plt.show()