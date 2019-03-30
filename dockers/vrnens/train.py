from __future__ import print_function
import argparse
import resource
import imp
import time
import logging
import math
import os
import pickle
import numpy as np
from path import Path
import theano
import theano.tensor as T
import lasagne

import sys
from config import get_config, add_to_config
sys.path.insert(0, '/vrnens')
from utils import checkpoints, metrics_logging
from collections import OrderedDict
from Logger import Logger
theano.config.reoptimize_unpickled_function = False
theano.config.cycle_detection = 'fast'  
#####################
# Training Functions#
#####################
#
# This function compiles all theano functions and returns
# two dicts containing the functions and theano variables.
#

def make_test_function(cfg, model, config):
    print("Compiling test function")
    X_ = T.TensorType('float32', [False]*5)('X')    
    y_ = T.TensorType('int32', [False]*1)('y')    
    l_out = model['l_out']

    # Batch Parameters
    batch_index_ = T.iscalar('batch_index_')
    batch_slice_ = slice(batch_index_*cfg['batch_size'], (batch_index_+1)*cfg['batch_size'])
    
    # Network Output
    y_hat_deterministic = lasagne.layers.get_output(l_out,X_,deterministic=True)

    # Sum every 24 examples
    raw_pred = T.sum(y_hat_deterministic,axis=0)
    pred = T.argmax(raw_pred)
    
    # Error rate
    classifier_test_error_rate = T.cast( T.mean( T.neq(pred, T.mean(y_,dtype='int32'))), 'float32' )
    classifier_loss = T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(y_hat_deterministic), y_)), 'float32')
    # Shared Variables
    X_shared_ = lasagne.utils.shared_empty(5, dtype='float32')

    y_shared_ = lasagne.utils.shared_empty(1, dtype='float32')

    # Test function
        
    test_error_fn = theano.function([batch_index_], [classifier_loss, classifier_test_error_rate ,pred, raw_pred, y_], givens={
            X_: X_shared_[batch_slice_],
            y_:  T.cast( y_shared_[batch_slice_], 'int32')      
        }) 
    tfuncs = {
             'test_function':test_error_fn,
            }
    tvars = {'X_' : X_,
             'y_' : y_,
             'X_shared_' : X_shared_,
             'y_shared_' : y_shared_,
             'batch_slice_' : batch_slice_,
             'batch_index_' : batch_index_
            }
    return tfuncs, tvars, model

def make_training_functions(cfg, model, config):
    print("Compiling training function")
    # Input Array
    X = T.TensorType('float32', [False]*5)('X')  
    
    # Shared Variable for input array
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
       
    # Class Vector
    y = T.TensorType('int32', [False]*1)('y')

    # Shared Variable for class vector
    y_shared = lasagne.utils.shared_empty(1, dtype='float32') 
    
    # Output layer
    l_out = model['l_out']

    # Batch Parameters
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    
    #####################################
    # Step 1: Compute full forward pass #
    ######################################
    
    # Get outputs
    y_hat = lasagne.layers.get_output(l_out,X) 

    #################################
    # Step 2: Define loss functions #
    #################################
    
    # L2 regularization for all params
    l2_all = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2)

    # Classifier loss function
    classifier_loss = T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(y_hat), y)), 'float32')
    
    # Classifier Error Rate
    classifier_error_rate = T.cast( T.mean( T.neq(T.argmax(y_hat,axis=1), y)), 'float32' )            

    # Regularized Loss 
    reg_loss = cfg['reg']*l2_all + classifier_loss
    
    # Get all network params
    params = lasagne.layers.get_all_params(l_out,trainable=True)
    
    # Handle annealing rate cases
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))
    
    ##########################
    # Step 3: Define Updates #
    ##########################
    max_rec = 0x100000
    resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
    sys.setrecursionlimit(max_rec)
    
    updates=lasagne.updates.nesterov_momentum(reg_loss,params,learning_rate=learning_rate)
    
    update_iter = theano.function([batch_index], [classifier_loss, classifier_error_rate],
            updates=updates, givens={
            X: X_shared[batch_slice],
            y:  T.cast( y_shared[batch_slice], 'int32')
        })

    tfuncs = {'update_iter':update_iter,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
            }
    return tfuncs, tvars, model

## Data augmentation function from Voxnet, which randomly translates
## and/or horizontally flips a chunk of data. Note that
def jitter_chunk(src, cfg,chunk_index):
    np.random.seed(chunk_index)
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst



# Main Function
def train(config):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    # Set random seed to ensure identical network initializations.
    # Note that cuDNN's convolutions are nondeterministic, so this
    # does not guarantee that two networks will behave identically.
    lasagne.random.set_rng(np.random.RandomState(1234))
                  
    # Load config file
    config_module = imp.load_source('config', config.model_definition)
    cfg = config_module.cfg
    # Get model
    model = config_module.get_model()
    # Compile functions
    print('Compiling theano functions...')
    test_function, test_vars, model = make_test_function(cfg, model, config)
    tfuncs, tvars, model = make_training_functions(cfg, model, config)
    tfuncs.update(test_function)
    tvars.update(test_vars)
    
    weights = config.weights
    if weights == -1:
        startepoch = 0
    else:
        ld = config.log_dir
        WEIGHTS = config.weights
        ckptfile = os.path.join(ld,config.snapshot_prefix+str(WEIGHTS)+'.npz')
        start_epoch = WEIGHTS + 1
        ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = WEIGHTS)
        LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)), os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = WEIGHTS)
        metadata = checkpoints.load_weights(ckptfile, model['l_out'])     
    print('Training...')
    itr = 0
    
    # Load data and shuffle training examples. 
    # Note that this loads the entire dataset into RAM! If you don't
    # have a lot of RAM, consider only loading chunks of this at a time.
    x_test = np.load(os.path.join(config.data, 'test.npz'))['features']
    y_test = np.load(os.path.join(config.data, 'test.npz'))['targets']
    x = np.load(os.path.join(config.data, 'train.npz'))['features']
    # Seed the shuffle
    np.random.seed(42)
    # Define shuffle indices
    index = np.random.permutation(len(x))
    # Shuffle inputs
    x = x[index]
    # Shuffle targets to match inputs
    y = np.load(os.path.join(config.data, 'train.npz'))['targets'][index]
    # Define size of chunk to be loaded into GPU memory
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    # Determine number of chunks
    num_chunks = int(math.ceil(len(y)/float(chunk_size)))
    print(num_chunks, chunk_size, num_chunks*chunk_size)
    # Get current learning rate
    new_lr = np.float32(tvars['learning_rate'].get_value())
    # Loop across training epochs!
    
    begin = start_epoch
    end = cfg['max_epochs']+start_epoch
    for epoch in xrange(begin, end + 1):    
        #EVAL
        evaluate(x_test, y_test, cfg, tfuncs, tvars, config, epoch=epoch)
        ACC_LOGGER.save(config.log_dir)
        LOSS_LOGGER.save(config.log_dir)
        ACC_LOGGER.plot(dest=config.log_dir)
        LOSS_LOGGER.plot(dest=config.log_dir)  
       
       # Update Learning Rate
        if isinstance(cfg['learning_rate'], dict) and epoch > 0:
            if any(x==epoch for x in cfg['learning_rate'].keys()):
                lr = np.float32(tvars['learning_rate'].get_value())
                new_lr = cfg['learning_rate'][epoch]
                print('Changing learning rate from {} to {}'.format(lr, new_lr))
                tvars['learning_rate'].set_value(np.float32(new_lr))
        if cfg['decay_rate'] and epoch > 0:
            lr = np.float32(tvars['learning_rate'].get_value())
            new_lr = lr*(1-cfg['decay_rate'])
            print('Changing learning rate from {} to {}'.format(lr, new_lr))
            tvars['learning_rate'].set_value(np.float32(new_lr))         
        
        # Loop across chunks!
        for chunk_index in xrange(num_chunks):
            # Define upper index of chunk to load
            # If you start doing complicated things with data loading, consider 
            # wrapping all of this into its own little function.
            upper_range = min(len(y),(chunk_index+1)*chunk_size)
            # Get current chunk
            x_shared = np.asarray(x[chunk_index*chunk_size:upper_range,:,:,:,:],dtype=np.float32)
            y_shared = np.asarray(y[chunk_index*chunk_size:upper_range],dtype=np.float32)
            # Get repeatable seed to shuffle jittered and unjittered instances within chunk.
            # Note that this seed varies between chunks, but will be constant across epochs.
            np.random.seed(chunk_index)
            # Get shuffled chunk indices for a second round of shuffling
            indices = np.random.permutation(2*len(x_shared))
            # Get number of batches in this chunk
            num_batches = 2*len(x_shared)//cfg['batch_size']
            
            # Combine data with jittered data, then shuffle and change binary range from {0,1} to {-1,3}, then load into GPU memory.
            tvars['X_shared'].set_value(4.0 * np.append(x_shared,jitter_chunk(x_shared, cfg,chunk_index),axis=0)[indices]-1.0, borrow=True)
            tvars['y_shared'].set_value(np.append(y_shared,y_shared,axis=0)[indices], borrow=True)

            lvs, accs = [],[]
            # Loop across batches!
            for bi in xrange(num_batches):
                
                [classifier_loss,class_acc] = tfuncs['update_iter'](bi)
                
                # Record batch loss and accuracy
                lvs.append(classifier_loss)
                accs.append(class_acc)
                
                # Update iteration counter
                itr += 1
                if it % max(config.train_log_frq/config.batch_size,1) == 0:
                    [closs,c_acc] = [float(np.mean(lvs)),1.0-float(np.mean(accs))]
                    ACC_LOGGER.log(c_acc,epoch,"train_accuracy")
                    LOSS_LOGGER.log(closs,epoch,"train_loss")  
                    lvs, accs = [],[] 
                    print('epoch: {0:^3d}, itr: {1:d}, c_loss: {2:.6f}, class_acc: {3:.5f}'.format(epoch, itr, closs, c_acc))       
                    
        if not (epoch % cfg['checkpoint_every_nth']) or epoch == end:
            weights_fname = os.path.join(config.log_dir,config.snapshot_prefix+str(epoch))
            checkpoints.save_weights(weights_fname, model['l_out'],
                                                {'itr': itr, 'ts': time.time(),
                                                'learning_rate': new_lr}) 

    print('training done')


def test(config):
    # Load config file
    config_module = imp.load_source('config', config.model_definition)
    cfg = config_module.cfg
    # Get model
    model = config_module.get_model()
    # Compile functions
    print('Compiling theano functions...')
    tfuncs, tvars, model = make_test_function(cfg, model, config)

    ckptfile = os.path.join(config.log_dir,config.snapshot_prefix+str(config.weights)+'.npz')
    metadata = checkpoints.load_weights(ckptfile, model['l_out'])
    x_t = np.load(os.path.join(config.data, 'test.npz'))['features']
    y_t = np.load(os.path.join(config.data, 'test.npz'))['targets']
    
    n_rotations = config.num_rotations
    confusion_matrix = np.zeros((40,40),dtype=np.int)
    num_test_batches = int(math.ceil(float(len(x_t))/float(n_rotations)))
    test_chunk_size = n_rotations*cfg['batches_per_chunk']
    num_test_chunks=int(math.ceil(float(len(x_t))/test_chunk_size))
    
    evaluate(x_t, y_t, cfg, tfuncs, tvars, config)

def evaluate(x_test, y_test, cfg, tfuncs, tvars, config, epoch=0):
    print("testing")
    n_rotations = cfg['n_rotations']
    chunk_size = n_rotations * cfg['batches_per_chunk']
    num_chunks = int(math.ceil(float(len(x_test))/chunk_size))
    
    labels = []
    test_class_error = []
    pred_array = []
    losses = []
    accs = []
    for chunk_index in xrange(num_chunks):
        upper_range = min(len(y_test),(chunk_index+1)*chunk_size) # 
        x_shared = np.asarray(x_test[chunk_index*chunk_size:upper_range,:,:,:,:],dtype=np.float32)
        y_shared = np.asarray(y_test[chunk_index*chunk_size:upper_range],dtype=np.float32)

        num_batches = int(math.ceil(float(len(x_shared))/n_rotations))
        
        tvars['X_shared_'].set_value(4.0 * x_shared-1.0, borrow=True)
        tvars['y_shared_'].set_value(y_shared, borrow=True)
        for bi in xrange(num_batches):
            [batch_loss, batch_test_class_error, pred, raw_pred, y] = tfuncs['test_function'](bi)
            losses.append(batch_loss)
            accs.append(batch_test_class_error)
            pred_array.append(np.array(pred))
            labels.append(y[0])
            
    loss, acc = [float(np.mean(losses)),1.0-float(np.mean(accs))]    
    predictions = list(pred_array)
    if not config.test:
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
    else:
        import Evaluation_tools as et
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir)  
    
if __name__=='__main__':
    config = get_config()
    if not config.test:
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        train(config)

        if config.weights == -1:
            config = add_to_config(config, 'weights', config.max_epoch)
        else:
            config = add_to_config(config, 'weights', config.max_epoch + config.weights)
        config = add_to_config(config, 'test', True)
    test(config)  
        

