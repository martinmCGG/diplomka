#!/usr/bin/env python
from __future__ import print_function
import caffe
from my_classify_modelnet import classify
from my_classify_modelnet import get_mean
import numpy as np
import os
from Logger import Logger

MAXEPOCH = 50
VIEWS = 12
PICTURES = 63660
TEST_PICTURES = 2468 * 12

def read_image(path, mean, net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    im = caffe.io.load_image(path)
    return transformer.preprocess('data', im)
    

def eval(args, solver, epoch=0):
    acc = 0
    loss = 0
    labels = []
    predictions = []
    
    test_file = os.path.join(args.data, 'testrotnet.txt')
    mean = get_mean(args.mean_file)
    
    batch_size = solver.test_nets[0].blobs['data'].num
    test_iters = int(TEST_PICTURES / batch_size)
    
    #files = [line.rstrip() for line in open(test_file)]
    
    for i in range(test_iters):
        #images = np.array([read_image(files[12*i+x].split()[0], mean, solver.test_nets[0]) for x in range(args.views)])
        #labels = [int(files[12*i+x].split()[1]) for x in range(args.views)]
        
        #solver.test_nets[0].blobs['data'] = images
        solver.test_nets[0].forward()
        #solver.test_nets[0].blobs['data'].data[...] = images
        #solver.test_nets[0].blobs['label'].data[...] = labels
        #out = solver.test_nets[0].forward()
        
        acc += solver.test_nets[0].blobs['my_accuracy'].data
        loss += solver.test_nets[0].blobs['(automatic)'].data
        
        probs = solver.test_nets[0].blobs['prob'].data
        predictions+= classify(probs)
        labels.append(int(solver.test_nets[0].blobs['label'].data[0]))
        
    acc /= test_iters
    loss  /= test_iters
    if not args.test:
        print("Accuracy: {:.3f}".format(acc))
        print("Loss: {:.3f}".format(loss))
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
    else:
        import Evaluation_tools as et
        eval_file = os.path.join(args.log_dir, 'rotnet.txt')
        et.write_eval_file(args.data, eval_file, predictions , labels , 'ROTNET')
        et.make_matrix(args.data, eval_file, args.log_dir)
        

def train(args, solver):
    
    if args.weights == -1:
        startepoch = 0
        solver.net.copy_from('./caffe_nets/ilsvrc13')
    else:
        weights = args.weights
        startepoch = weights + 1
        snapshot = os.path.join(args.log_dir, 'case1_iter_'+str(weights))
        print(snapshot)
        ACC_LOGGER.load((os.path.join(args.log_dir,"rotnet_acc_train_accuracy.csv"),os.path.join(args.log_dir,"rotnet_acc_eval_accuracy.csv")), epoch = weights)
        LOSS_LOGGER.load((os.path.join(args.log_dir,"rotnet_loss_train_loss.csv"), os.path.join(args.log_dir,'rotnet_loss_eval_loss.csv')), epoch = weights)      
        solver.restore(snapshot + '.solverstate')
        solver.net.copy_from(snapshot + '.caffemodel')
        solver.test_nets[0].copy_from(snapshot + '.caffemodel')
        
    steps_per_epoch = PICTURES/ solver.net.blobs['data'].data.shape[0] + 1
    print(steps_per_epoch)

    for epoch in range(startepoch, startepoch+MAXEPOCH):
        eval(args, solver, epoch=epoch)    
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
    parser.add_argument("--data", default='/data/converted', help="path to dataset")
    parser.add_argument("--solver", required=True, help="Solver proto definition.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of device ids.")
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--mean_file', default="/opt/caffe/caffe-rotationnet2/data/ilsvrc12/imagenet_mean.binaryproto", type=str)
    parser.add_argument('--views', default=12, type=int)
    parser.add_argument("--test", action='store_true')
    parser.add_argument('--weights', default=-1, type=int)

    args = parser.parse_args()
    caffe.set_device(args.gpus[0])
    caffe.set_mode_gpu()
    solver = caffe.get_solver(args.solver)
    
    
    if not args.test:
        LOSS_LOGGER = Logger("rotnet_loss")
        ACC_LOGGER = Logger("rotnet_acc")
        train(args, solver)
    else:
        weights = args.weights
        snapshot = os.path.join(args.log_dir, 'case1_iter_'+str(weights))  
        solver.restore(snapshot + '.solverstate')
        solver.net.copy_from(snapshot + '.caffemodel')
        solver.test_nets[0].copy_from(snapshot + '.caffemodel')
        eval(args, solver)
        