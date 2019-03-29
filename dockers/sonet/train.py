from __future__ import print_function
import copy
import numpy as np
import math
import os
import sys
sys.path.append('/sonet')
sys.path.append('/sonet/util')

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from Logger import Logger
from config import get_config
from models.classifier import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader

def train(model, config):
    
    trainset = ModelNet_Shrec_Loader(os.path.join(config.data, 'train_files.txt'), 'train', config.data, config)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_threads)
    print('#training point clouds = %d' % len(trainset))
    
    start_epoch = 0
    WEIGHTS = config.weights
    if WEIGHTS!=-1:
        ld = config.log_dir
        start_epoch = WEIGHTS + 1
        ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),
                         os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = WEIGHTS)
        LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)),
                           os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = WEIGHTS)
        
    print("Starting training")
    best_accuracy = 0
    losses = []
    accs = []
    if config.num_classes == 10:
        config.dropout = config.dropout + 0.1
    for epoch in range(start_epoch, config.max_epoch + start_epoch + 1):
        epoch_iter = 0
        for i, data in enumerate(trainloader):
            epoch_iter += config.batch_size

            input_pc, input_sn, input_label, input_node, input_node_knn_I = data
            model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
            
            model.optimize(epoch=epoch)
            errors = model.get_current_errors()            
            losses.append(errors['train_loss'])
            accs.append(errors['train_accuracy'])
            
            if i % max(config.train_log_frq/config.batch_size,1) == 0:
                acc = np.mean(accs)
                loss = np.mean(losses) 
                LOSS_LOGGER.log(loss, epoch, "train_loss")
                ACC_LOGGER.log(acc, epoch, "train_accuracy")
                print("EPOCH {} acc: {} loss: {}".format(epoch, acc, loss))
                ACC_LOGGER.save(config.log_dir)
                LOSS_LOGGER.save(config.log_dir)
                ACC_LOGGER.plot(dest=config.log_dir)
                LOSS_LOGGER.plot(dest=config.log_dir)
                losses = []
                accs = []

        best_accuracy = test(model, config, best_accuracy=best_accuracy, epoch=epoch)

        if epoch % config.save_each == 0:
            print("Saving network...")
            save_path = os.path.join(config.log_dir,config.snapshot_prefix +'_encoder_'+str(epoch))
            model.save_network(model.encoder, save_path, 0)
            save_path = os.path.join(config.log_dir,config.snapshot_prefix +'_classifier_'+str(epoch))
            model.save_network(model.classifier, save_path, 0)

        
        if config.num_classes == 10:
            lr_decay_step = 40
        else:
            lr_decay_step = 20
        if epoch%lr_decay_step==0 and epoch > 0:
            model.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (config.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % config.bn_momentum_decay_step == 0):
            current_bn_momentum = config.bn_momentum * (
            config.bn_momentum_decay ** (next_epoch // config.bn_momentum_decay_step))
            print('BN momentum updated to: %f' % current_bn_momentum)
            
def test(model, config, best_accuracy=0, epoch=None):
    batch_amount = 0
    model.test_loss.data.zero_()
    model.test_accuracy.data.zero_()
    
    predictions = []
    labels = []
    
    for i, data in enumerate(testloader):
        input_pc, input_sn, input_label, input_node, input_node_knn_I = data
        model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
        model.test_model()

        batch_amount += input_label.size()[0]
        model.test_loss += model.loss.detach() * input_label.size()[0]

        # accumulate accuracy
        _, predicted_idx = torch.max(model.score.data, dim=1, keepdim=False)
        predictions += list(predicted_idx)
        labels += list(input_label)
        
        correct_mask = torch.eq(predicted_idx, model.input_label).float()
        test_accuracy = torch.mean(correct_mask).cpu()
        
        model.test_accuracy += test_accuracy * input_label.size()[0]
        
    model.test_loss /= batch_amount
    model.test_accuracy /= batch_amount

    if config.test:
        predictions = [x.item() for x in predictions]
        labels = [x.item() for x in labels]
        #print(sum([0 if abs(predictions[i] - labels[i]) else 1 for i in range(len(predictions))]) / 2648.0)
        import Evaluation_tools as et
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir)    
    else:
        if model.test_accuracy.item() > best_accuracy:
            best_accuracy = model.test_accuracy.item()
        loss = model.test_loss.item()
        acc = model.test_accuracy.item()
        print('Tested network. So far best: %f' % best_accuracy)
        print("TESTING EPOCH {} acc: {} loss: {}".format(epoch, acc, loss ))       
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
        return best_accuracy

if __name__=='__main__':
    
    config = get_config()
    LOSS_LOGGER = Logger("{}_loss".format(config.name))
    ACC_LOGGER = Logger("{}_acc".format(config.name))
    testset = ModelNet_Shrec_Loader(os.path.join(config.data, 'test_files.txt'), 'test',config.data, config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_threads)
    
    model = Model(config)
    if config.weights != -1:
        weights = os.path.join(config.log_dir,config.snapshot_prefix +'_encoder_'+str(config.weights))
        model.encoder.load_state_dict(torch.load(weights))
        weights = os.path.join(config.log_dir,config.snapshot_prefix +'_classifier_'+str(config.weights))
        model.classifier.load_state_dict(torch.load(weights))

    if config.test:
        test(model, config)
    else:
        train(model, config)


            


