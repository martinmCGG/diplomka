from __future__ import print_function

import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
import modelnet_h5_dataset
from Logger import Logger
from config import get_config
config = get_config()

MODEL = importlib.import_module(config.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', config.model+'.py')
if not os.path.exists(config.log_dir): os.mkdir(config.log_dir)
os.system('cp %s %s' % ('config.ini', config.log_dir)) # bkp of model def
LOG_FOUT = open(os.path.join(config.log_dir, 'log_train.txt'), 'w')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(config.decay_step)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 40


# Shapenet official train/test split
if config.normal:
    CHANNELS=6
    assert(config.num_points<=2048)
    """TRAIN_FILES = os.path.join(config.data, 'train_files.txt')
    TEST_FILES = os.path.join(config.data, 'test_files.txt')
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(TRAIN_FILES, batch_size=config.batch_size, npoints=config.num_points, shuffle=True, normal_channel=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(TEST_FILES, batch_size=config.batch_size, npoints=config.num_points, shuffle=False, normal_channel=True)"""
    assert(config.num_points<=10000)
    DATA_PATH = os.path.join(config.data, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=config.num_points, split='train', normal_channel=config.normal, batch_size=config.batch_size)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=config.num_points, split='test', normal_channel=config.normal, batch_size=config.batch_size)
else:
    assert(config.num_points<=2048)
    CHANNELS=3
    TRAIN_FILES = os.path.join(config.data, 'train_files.txt')
    TEST_FILES = os.path.join(config.data, 'test_files.txt')
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(TRAIN_FILES, batch_size=config.batch_size, npoints=config.num_points, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(TEST_FILES, batch_size=config.batch_size, npoints=config.num_points, shuffle=False)
    
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        config.learning_rate,  # Base learning rate.
                        batch * config.batch_size,  # Current index into the dataset.
                        config.decay_step,          # Decay step.
                        config.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*config.batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(config):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(config.batch_size, config.num_points)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl,NUM_CLASSES=NUM_CLASSES, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')

            print ("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            if config.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=config.momentum)
            elif config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=1000)
        
        # Create a session
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        tf_config.log_device_placement = False
        sess = tf.Session(config=tf_config)

        
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        start_epoch = 0   
        if config.weights != -1:
            WEIGHTS = config.weights
            ld = config.log_dir
            ckptfile = os.path.join(ld,config.snapshot_prefix+str(WEIGHTS))
            saver.restore(sess, ckptfile)
            start_epoch = config.weights + 1
            ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = WEIGHTS)
            LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)), os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = WEIGHTS)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(start_epoch, start_epoch+config.max_epoch + 1):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            eval_one_epoch(config,sess, ops, epoch = epoch)             
            train_one_epoch(config,sess, ops, epoch = epoch)
            
            ACC_LOGGER.save(config.log_dir)
            LOSS_LOGGER.save(config.log_dir)
            ACC_LOGGER.plot(dest=config.log_dir)
            LOSS_LOGGER.plot(dest=config.log_dir)
            
            # Save the variables to disk.
            if epoch % config.save_period == 0:
                checkpoint_path = os.path.join(config.log_dir, config.snapshot_prefix+str(epoch))
                saver.save(sess, checkpoint_path)



def train_one_epoch(config,sess, ops, epoch=0):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((config.batch_size,config.num_points,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((config.batch_size), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
        #batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,}
        step, _, loss_val, pred_val = sess.run([ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        
        
        if (batch_idx+1)%20 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 200))
            LOSS_LOGGER.log((loss_sum / 200), epoch, "train_loss")
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            ACC_LOGGER.log((total_correct / float(total_seen)), epoch, "train_accuracy")
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()

def test(config):
    is_training = False
     
    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(config.batch_size, config.num_points)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    tf_config.log_device_placement = False
    sess = tf.Session(config=tf_config)

    ld = config.log_dir
    ckptfile = os.path.join(ld,config.snapshot_prefix+str(config.weights))
    saver.restore(sess, ckptfile)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss}
    eval_one_epoch(config,sess, ops)

def eval_one_epoch(config, sess, ops, topk=1, epoch=0):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((config.batch_size,config.num_points,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((config.batch_size), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []

    predictions = []
    labels = []

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_pred_sum = np.zeros((config.batch_size, config.num_classes)) # score for classes
        for vote_idx in range(config.num_votes):
            # Shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(config.num_points)
            np.random.shuffle(shuffled_indices)
            if config.normal:
                rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:, shuffled_indices, :],
                    vote_idx/float(config.num_votes) * np.pi * 2)
            else:
                rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                    vote_idx/float(config.num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            batch_pred_sum += pred_val
        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        
        predictions += pred_val[0:bsize].tolist()
        labels += batch_label[0:bsize].tolist()
        
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    if config.test:
        import Evaluation_tools as et
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir)
    else:
        log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
        LOSS_LOGGER.log((loss_sum / float(len(TEST_FILES))),epoch, "eval_loss")
        log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
        ACC_LOGGER.log((total_correct / float(total_seen)),epoch, "eval_accuracy")
        TEST_DATASET.reset()
        return total_correct/float(total_seen)

if __name__ == "__main__":
    if config.test:
        test(config)        
    else:
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        train(config)
        LOG_FOUT.close()

