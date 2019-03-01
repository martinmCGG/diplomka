#!/usr/bin/env python
from __future__ import print_function
import caffe
import numpy as np
import os
import lmdb
from Logger import Logger

MAXEPOCH = 40

def eval(args, solver, epoch=0):
    acc = 0
    loss = 0
    labels = []
    predictions = []
    
    test_file = os.path.join(args.data, 'test.txt')
    with open(test_file, 'r') as f:
        test_count = len(f.readlines())
    keys = solver.test_nets[0].blobs.keys()
    print(keys)
    if 'label_octreedatabase_1_split_0' in keys:
        batch_size = (solver.test_nets[0].blobs['label_octreedatabase_1_split_0'].data.shape[0])
    else:
        batch_size = (solver.test_nets[0].blobs['label_data_1_split_0'].data.shape[0])
        
    test_iters = test_count / batch_size

    for i in range(test_iters):
        
        solver.test_nets[0].forward()

        acc += solver.test_nets[0].blobs['accuracy'].data
        loss += solver.test_nets[0].blobs['loss'].data
        
        probs = solver.test_nets[0].blobs['ip2'].data
        predictions += list(np.argmax(np.array(probs), axis=1))
        labels += list(solver.test_nets[0].blobs['label'].data)
        
    solver.test_nets[0].forward()
    acc += solver.test_nets[0].blobs['accuracy'].data
    loss += solver.test_nets[0].blobs['loss'].data
    probs = solver.test_nets[0].blobs['ip2'].data
    predictions += list(np.argmax(np.array(probs), axis=1))[0:test_count%batch_size]
    labels += list(solver.test_nets[0].blobs['label'].data)[0:test_count%batch_size]
        
    acc /= test_iters + 1
    loss  /= test_iters + 1
    
    if not args.test:
        print("Accuracy: {:.3f}".format(acc))
        print("Loss: {:.3f}".format(loss))
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
    else:
        import Evaluation_tools as et
        labels = [int(l) for l in labels]
        eval_file = os.path.join(args.log_dir, 'octree.txt')
        et.write_eval_file(args.data, eval_file, predictions , labels , 'OCTREE')
        et.make_matrix(args.data, eval_file, args.log_dir)
        

def train(args, solver):
    
    if args.weights == -1:
        startepoch = 0

    else:
        weights = args.weights
        startepoch = weights + 1
        snapshot = os.path.join(args.log_dir, 'M40_iter_'+str(weights))
        print(snapshot)
        ACC_LOGGER.load((os.path.join(args.log_dir,"octree_acc_train_accuracy.csv"),os.path.join(args.log_dir,"octree_acc_eval_accuracy.csv")), epoch = weights)
        LOSS_LOGGER.load((os.path.join(args.log_dir,"octree_loss_train_loss.csv"), os.path.join(args.log_dir,'octree_loss_eval_loss.csv')), epoch = weights)      
        solver.restore(snapshot + '.solverstate')
        solver.net.copy_from(snapshot + '.caffemodel')
        solver.test_nets[0].copy_from(snapshot + '.caffemodel')
        
    steps_per_epoch = 4000

    for epoch in range(startepoch, startepoch+MAXEPOCH):
        eval(args, solver, epoch=epoch)            
        losses = []
        accs = []
        for it in range(steps_per_epoch):
            solver.step(1)
            loss = float(solver.net.blobs['loss'].data)
            acc = float(solver.net.blobs['accuracy'].data)
            losses.append(loss)
            accs.append(acc)
            
            if it%100 == 0:
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
    parser.add_argument("--data", default='/data', help="path to dataset")
    parser.add_argument("--solver", required=True, help="Solver proto definition.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of device ids.")
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--views', default=12, type=int)
    parser.add_argument("--test", action='store_true')
    parser.add_argument('--weights', default=-1, type=int)

    args = parser.parse_args()
    caffe.set_device(args.gpus[0])
    caffe.set_mode_gpu()
    solver = caffe.get_solver(args.solver)
    
    if not args.test:
        LOSS_LOGGER = Logger("octree_loss")
        ACC_LOGGER = Logger("octree_acc")
        train(args, solver)
    else:
        weights = args.weights
        snapshot = os.path.join(args.log_dir, 'M40_iter_'+str(weights))  
        solver.restore(snapshot + '.solverstate')
        solver.net.copy_from(snapshot + '.caffemodel')
        solver.test_nets[0].copy_from(snapshot + '.caffemodel')
        eval(args, solver)
        