from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", default=32, type=int, help="Resolution of the voxel grid")
    parser.add_argument("-f", default= "D:\\workspace\\diplomka\\src\\3Dconversion\\test.npz", type=str, help="npz file to load")
    parser.add_argument("-r", default = 24, type=int, help="Number of rotations of model along vertical axis")
    args = parser.parse_args()

    
    xt = np.asarray(np.load(args.f)['features'],dtype=np.float32)
    yt = np.asarray(np.load(args.f)['targets'],dtype=np.float32)
    N = args.v
    index = 0
    ma = xt[index].reshape((N,N,N))
    print(np.count_nonzero(xt[0] - xt[1]))
    
    print(xt.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    ax.voxels(ma)
    
    plt.show()