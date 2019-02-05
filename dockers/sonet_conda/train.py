from __future__ import print_function
import copy
import numpy as np
import math
import os
import sys
sys.path.append('/sonet')
sys.path.append('/sonet/util')
from options import Options
os.system('export CUDA_VISIBLE_DEVICES=0')
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from Logger import Logger

from models.classifier import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
#from util.visualizer import Visualizer


if __name__=='__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',default=None, type=int, help='Number of model to finetune or evaluate')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()    
    opt.weights = args.weights
    trainset = ModelNet_Shrec_Loader(os.path.join(opt.data, 'train_files.txt'), 'train', opt.data, opt)
    
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    print('#training point clouds = %d' % len(trainset))

    testset = ModelNet_Shrec_Loader(os.path.join(opt.data, 'test_files.txt'), 'test',opt.data, opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)

    start_epoch = 0
    if not args.test:
        LOSS_LOGGER = Logger("sonet_loss")
        ACC_LOGGER = Logger("sonet_acc")
    
    # create model, optionally load pre-trained model
    model = Model(opt)
    if args.weights is not None:
        start_epoch = args.weights + 1
        weights = 'encoder_{}'.format(args.weights)
        model.encoder.load_state_dict(torch.load(weights))
        weights = 'classifier_{}'.format(args.weights)
        model.classifier.load_state_dict(torch.load(weights))
        if args.test:
            ACC_LOGGER.load((os.path.join(args.log_dir,"sonet_acc_train_accuracy.csv"),os.path.join(args.log_dir,"sonet_acc_eval_accuracy.csv")), epoch=weights)
            LOSS_LOGGER.load((os.path.join(args.log_dir,"sonet_loss_train_loss.csv"), os.path.join(args.log_dir,'sonet_loss_eval_loss.csv')), epoch=weights)
    ############################# automation for ModelNet10 / 40 configuration ####################
    if opt.classes == 10:
        opt.dropout = opt.dropout + 0.1
    ############################# automation for ModelNet10 / 40 configuration ####################

    
    print("Starting training")
    best_accuracy = 0
    losses = []
    accs = []
    for epoch in range(start_epoch, opt.max_epoch + start_epoch):
        
        epoch_iter = 0
        for i, data in enumerate(trainloader):
            epoch_iter += opt.batch_size

            input_pc, input_sn, input_label, input_node, input_node_knn_I = data
            model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
            
            model.optimize(epoch=epoch)
            errors = model.get_current_errors()            
            losses.append(errors['train_loss'])
            accs.append(errors['train_accuracy'])
            
            if i % 200 == 0:

                acc = np.mean(accs)
                loss = np.mean(losses) 
                LOSS_LOGGER.log(loss, epoch, "train_loss")
                ACC_LOGGER.log(acc, epoch, "train_accuracy")
                print("EPOCH {} acc: {} loss: {}".format(epoch, acc, loss))
                ACC_LOGGER.save(opt.log_dir)
                LOSS_LOGGER.save(opt.log_dir)
                ACC_LOGGER.plot(dest=opt.log_dir)
                LOSS_LOGGER.plot(dest=opt.log_dir)
                losses = []
                accs = []

        if epoch % opt.save_each == 0:
            print("Saving network...")
            model.save_network(model.encoder, 'encoder', '%d' % (epoch), opt.gpu_id)
            model.save_network(model.classifier, 'classifier', '%d' % (epoch), opt.gpu_id)

        # test network
    
        batch_amount = 0
        model.test_loss.data.zero_()
        model.test_accuracy.data.zero_()
        for i, data in enumerate(testloader):
            input_pc, input_sn, input_label, input_node, input_node_knn_I = data
            model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
            model.test_model()

            batch_amount += input_label.size()[0]

            # # accumulate loss
            model.test_loss += model.loss.detach() * input_label.size()[0]

            # # accumulate accuracy
            _, predicted_idx = torch.max(model.score.data, dim=1, keepdim=False)
            correct_mask = torch.eq(predicted_idx, model.input_label).float()
            test_accuracy = torch.mean(correct_mask).cpu()
            model.test_accuracy += test_accuracy * input_label.size()[0]

        model.test_loss /= batch_amount
        model.test_accuracy /= batch_amount
        if model.test_accuracy.item() > best_accuracy:
            best_accuracy = model.test_accuracy.item()
            
        loss = model.test_loss.item()
        acc = model.test_accuracy.item()
        print('Tested network. So far best: %f' % best_accuracy)
        print("TESTING EPOCH {} acc: {} loss: {}".format(epoch, acc, loss ))       
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
            

        # learning rate decay
        if opt.classes == 10:
            lr_decay_step = 40
        else:
            lr_decay_step = 20
        if epoch%lr_decay_step==0 and epoch > 0:
            model.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (opt.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % opt.bn_momentum_decay_step == 0):
            current_bn_momentum = opt.bn_momentum * (
            opt.bn_momentum_decay ** (next_epoch // opt.bn_momentum_decay_step))
            print('BN momentum updated to: %f' % current_bn_momentum)






