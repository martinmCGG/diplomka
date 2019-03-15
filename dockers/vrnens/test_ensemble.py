import argparse
import imp
import time
import logging
import math
import os

import numpy as np
from path import Path
import theano
import theano.tensor as T
import lasagne

import sys
sys.path.insert(0, '/vrnens')
from utils import checkpoints,metrics_logging
from collections import OrderedDict

# Define the training functions
def make_training_functions(cfg, model, args):
    
    test_function_file = os.path.join(args.log_dir,"test_function")    
    # Inputs
    X = T.TensorType('float32', [False]*5)('X')    

    # Y for test classification
    y = T.TensorType('int32', [False]*1)('y')    
    
    # Output layer
    l_out = model['l_out']

    # Batch Parameters
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    test_batch_slice = slice(batch_index*24, (batch_index+1)*24)
    test_batch_i = batch_index
    
    # Network Output
    y_hat_deterministic = lasagne.layers.get_output(l_out,X,deterministic=True)

    # Sum every 24 examples
    raw_pred = T.sum(y_hat_deterministic,axis=0)
    pred = T.argmax(raw_pred)
    
    # Error rate
    classifier_test_error_rate = T.cast( T.mean( T.neq(pred, T.mean(y,dtype='int32'))), 'float32' )
    classifier_loss = T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(y_hat_deterministic), y)), 'float32')
    # Shared Variables
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')

    y_shared = lasagne.utils.shared_empty(1, dtype='float32')

    # Test function
        
    test_error_fn = theano.function([batch_index], [classifier_loss, classifier_test_error_rate ,pred, raw_pred, y], givens={
            X: X_shared[batch_slice],
            y:  T.cast( y_shared[batch_slice], 'int32')      
        }) 
    tfuncs = {
             'test_function':test_error_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index
            }
    return tfuncs, tvars, model

def test(args):
    
    config_file =args.model
    lasagne.random.set_rng(np.random.RandomState(0))
    print(config_file)
    config_module = imp.load_source('config', config_file)
    cfg = config_module.cfg
   
    weights_fname = os.path.join(args.log_dir, 'model.ckpt-'+str(args.weights)+'.npz')

    model = config_module.get_model()
    print('Compiling theano functions...')
    tfuncs, tvars, model = make_training_functions(cfg, model, args)
    metadata = checkpoints.load_weights(weights_fname, model['l_out'])

    print('Testing...')
    
    # Get Test Data 
    x_t = np.load(os.path.join(args.data, 'test.npz'))['features']
    y_t = np.load(os.path.join(args.data, 'test.npz'))['targets']

    n_rotations = 24  
    confusion_matrix = np.zeros((40,40),dtype=np.int)
    num_test_batches = int(math.ceil(float(len(x_t))/float(n_rotations)))
    test_chunk_size = n_rotations*cfg['batches_per_chunk']
    num_test_chunks=int(math.ceil(float(len(x_t))/test_chunk_size))
  
    test_class_error = []
    pred_array = []
    test_itr=0
    
    predictions = []
    labels = []
    pred_array = []
    # Evaluate on test set
    for chunk_index in xrange(num_test_chunks):
        upper_range = min(len(y_t),(chunk_index+1)*test_chunk_size) # 
        x_shared = np.asarray(x_t[chunk_index*test_chunk_size:upper_range,:,:,:,:],dtype=np.float32)
        y_shared = np.asarray(y_t[chunk_index*test_chunk_size:upper_range],dtype=np.float32)

        num_batches = int(math.ceil(float(len(x_shared))/n_rotations))
        tvars['X_shared'].set_value(4.0 * x_shared-1.0, borrow=True)
        tvars['y_shared'].set_value(y_shared, borrow=True)
        lvs, accs = [],[]      
        for bi in xrange(num_batches):
            [classifier_loss, test_error_rate ,pred, raw_pred, y] =  tfuncs['test_function'](bi)
    
            test_class_error.append(test_error_rate)
            pred_array.append(np.array(pred))
            labels.append(y[0])
            
    print('Test acc is: ' + str(1 - np.mean(test_class_error)))
    return pred_array, labels


def evaluate_ensemble(all_preds, labels):
    all_preds = np.array(all_preds)
    summed_preds = np.sum(all_preds, axis = 0)
    final_predictions = np.argmax(summed_preds, 1)
    return final_predictions
        

### TODO: Clean this up and add the necessary arguments to enable all of the options we want.
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to file containing model definition')
    parser.add_argument('data', default="/data/converted", help='path to data folder')
    parser.add_argument('--log_dir', default="logs", help='path to data folder')
    parser.add_argument('--weights', type=int, help='number of model to test')
    args = parser.parse_args()
    file = args.model
    
    predictions, labels = test(args)
    
    import Evaluation_tools as et
    eval_file = os.path.join(args.log_dir, 'vrnens.txt')
    et.write_eval_file(args.data, eval_file, predictions, labels, 'VRNENS')
    et.make_matrix(args.data, eval_file, args.log_dir)
