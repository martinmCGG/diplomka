import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default= "D:\\workspace\\diplomka\\src\\3Dconversion\\val_0_0.h5", type=str, help="npz file to load")
    args = parser.parse_args()
    
    f = h5py.File(args.f, 'r')
    index = 50
    labels = np.array(f['label'])
    print(labels[index])
    points = np.array(f['data'])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[index,:,0], points[index,:,1],points[index,:,2])
    
    plt.show()