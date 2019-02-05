#!/usr/bin/env python
from __future__ import print_function
import caffe
from my_classify_modelnet import classify
import numpy as np

MAXEPOCH = 50
VIEWS = 12
PICTURES = 63660

def train(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        gpus,  # list of device ids
):
    gpus=[0]
    caffe.set_device(gpus[0])
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver)
    if snapshot and len(snapshot) != 0:
        solver.restore(snapshot)
    solver.net.copy_from('./caffe_nets/ilsvrc13')
    solver.solve()
    #solver.step(10000)
    '''print(solver.net.blobs)
    print(solver.test_nets[0].blobs)
    
    steps_per_epoch = 63660 / solver.net.blobs['data'].data.shape[0] + 1
    print(steps_per_epoch)
    for epoch in range(MAXEPOCH):
        losses = []
        for step in range(steps_per_epoch):
            solver.step(1)
            loss = solver.net.blobs['(automatic)'].data
            print(loss)
            losses.append(loss)
        print("LOSSES: ", np.mean(losses))'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--solver", required=True, help="Solver proto definition.")
    parser.add_argument("--snapshot", help="Solver snapshot to restore.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0],
                        help="List of device ids.")

    args = parser.parse_args()

    train(args.solver, args.snapshot, args.gpus)