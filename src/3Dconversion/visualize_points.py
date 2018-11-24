import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default= "D:\\workspace\\diplomka\\src\\3Dconversion\\model_normalized.objpcl.npy", type=str, help="npz file to load")
    args = parser.parse_args()

    points = np.load(args.f)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:,0], points[:,1],points[:,2])
    
    plt.show()