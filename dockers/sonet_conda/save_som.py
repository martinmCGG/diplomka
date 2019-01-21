#!/usr/bin/env python
# coding: utf-8

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import h5py
import json

from util import som

# for train and val
def som_saver_shrec2016(root, rows, cols, gpu_ids, output_root):
    som_builder = som.SOM(rows, cols, 3, gpu_ids)
    
    folder_list = os.listdir(root)
    for i, folder in enumerate(folder_list):
        file_list = os.listdir(os.path.join(root, folder))
        for j, file in enumerate(file_list):
            if file[-3:] == 'txt':
                data = np.loadtxt(os.path.join(root, folder, file))


                npz_file = os.path.join(output_root, file[0:-4]+'.npz')
                np.savez(npz_file, pc=pc_np, sn=sn_np, som_node=som_node_np)

                if j%100==0:
                    print('%s, %s' % (folder, file))


def som_one_cloud(data):
    pc_np = data[:, 0:3]
    sn_np = data[:, 3:6]
    
    pc_np_sampled = pc_np[np.random.choice(pc_np.shape[0], 2048, replace=False), :]
    pc = torch.from_numpy(pc_np_sampled.transpose().astype(np.float32)).cuda()  # 3xN tensor
    som_builder.optimize(pc)
    som_node_np = som_builder.node.cpu().numpy().transpose().astype(np.float32)  # node_numx3
    return som_node_np
    


rows, cols = 8, 8
som_saver_shrec2016('/ssd/dataset/SHREC2016/obj_txt/test_allinone', rows, cols, True, '/ssd/dataset/SHREC2016/%dx%d/test'%(rows, cols))


file = '/ssd/dataset/SHREC2016/8x8/train/model_013435.npz'
data = np.load(file)
pc_np = data['pc']
sn_np = data['sn']
som_node_np = data['som_node']

print(pc_np)
print(sn_np)
print(som_node_np)

x_np = pc_np
node_np = som_node_np
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_np[:,0].tolist(), x_np[:,1].tolist(), x_np[:,2].tolist(), s=1)
ax.scatter(node_np[:,0].tolist(), node_np[:,1].tolist(), node_np[:,2].tolist(), s=6, c='r')
plt.show()






