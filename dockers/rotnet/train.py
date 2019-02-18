#!/usr/bin/env python
from __future__ import print_function
import caffe
from my_classify_modelnet import classify
import numpy as np
import os
from Logger import Logger

MAXEPOCH = 50
VIEWS = 12
PICTURES = 63660
TEST_PICTURES = 2468 * 12

def eval(solver, epoch):
    acc = 0
    loss = 0

    batch_size = solver.test_nets[0].blobs['data'].num
    print(solver.test_nets[0].blobs.keys())
    test_iters = int(TEST_PICTURES / batch_size)
    print(batch_size, test_iters)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        acc += solver.test_nets[0].blobs['my_accuracy'].data
        loss+= solver.test_nets[0].blobs['(automatic)'].data

        
    acc /= test_iters
    loss  /= test_iters
    
    print("Accuracy: {:.3f}".format(acc))
    print("Loss: {:.3f}".format(loss))
    LOSS_LOGGER.log(loss, epoch, "eval_loss")
    ACC_LOGGER.log(acc, epoch, "eval_accuracy")
    

def train(args):
    solver = args.solver
    gpus = args.gpus
    
    gpus=[0]
    caffe.set_device(gpus[0])
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver)
    if args.weights == -1:
        startepoch = 0
        solver.net.copy_from('./caffe_nets/ilsvrc13')
    else:
        weights = args.weights
        startepoch = weights + 1
        snapshot = os.path.join(args.log_dir, 'case1_iter_'+str(weights) + '.solverstate')
        ACC_LOGGER.load((os.path.join(args.log_dir,"rotnet_acc_train_accuracy.csv"),os.path.join(args.log_dir,"rotnet_acc_eval_accuracy.csv")), epoch = weights)
        LOSS_LOGGER.load((os.path.join(args.log_dir,"rotnet_loss_train_loss.csv"), os.path.join(args.log_dir,'rotnet_loss_eval_loss.csv')), epoch = weights)      
        solver.restore(snapshot)
    
    steps_per_epoch = PICTURES/ solver.net.blobs['data'].data.shape[0] + 1
    print(steps_per_epoch)

    for epoch in range(startepoch, startepoch+MAXEPOCH):
        eval(solver, epoch)    
        losses = []
        accs = []
        for it in range(steps_per_epoch):
            
            solver.step(1)
            loss = solver.net.blobs['(automatic)'].data
            acc = solver.net.blobs['my_accuracy'].data
            losses.append(loss)
            accs.append(acc)
            
            if it%20 == 0:
                LOSS_LOGGER.log(np.mean(losses), epoch, "train_loss")
                ACC_LOGGER.log(np.mean(accs), epoch, "train_accuracy")
                ACC_LOGGER.save(args.log_dir)
                LOSS_LOGGER.save(args.log_dir)
                losses = []
                accs = []
                
        ACC_LOGGER.plot(dest=args.log_dir)
        LOSS_LOGGER.plot(dest=args.log_dir)        
        print("LOSS: ", np.mean(losses))
        print("ACCURACY", np.mean(accs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--solver", required=True, help="Solver proto definition.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of device ids.")
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--views', default=12, type=int)
    parser.add_argument("--test", action='store_true')
    parser.add_argument('--weights', default=-1, type=int)
    args = parser.parse_args()

    LOSS_LOGGER = Logger("rotnet_loss")
    ACC_LOGGER = Logger("rotnet_acc")

    train(args)