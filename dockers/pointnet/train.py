import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
from Logger import Logger
from config import get_config, add_to_config

config = get_config()

MODEL = importlib.import_module(config.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', config.model+'.py')
LOG_DIR = config.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % ('config.ini', config.log_dir)) # bkp of model def
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(config)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_RATE = 0.5
BN_DECAY_STEP = float(config.decay_step)
BN_DECAY_CLIP = 0.99

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(config.data,'train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(config.data, 'test_files.txt'))
print(TEST_FILES)
WEIGHTS = config.weights


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
                      BN_DECAY_STEP,
                      BN_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(config):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(config.batch_size, config.num_points)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES=config.num_classes, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(config.batch_size)

            # Get training operator
            learning_rate = get_learning_rate(batch)

            if config.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=config.momentum)
            elif config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=15)
            
        # Create a session
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        tf_config.log_device_placement = False
        sess = tf.Session(config=tf_config)
        
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})
        
        # Init variables
        start_epoch = 0
        
        if WEIGHTS!=-1:
            ld = config.log_dir
            ckptfile = os.path.join(ld,config.snapshot_prefix+str(WEIGHTS))
            saver.restore(sess, ckptfile)
            start_epoch = WEIGHTS + 1
            ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),
                             os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = WEIGHTS)
            LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)),
                               os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = WEIGHTS)
          
            
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'step': batch}
        
        begin = start_epoch
        end = config.max_epoch+start_epoch
        for epoch in range(begin, end+1):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            eval_one_epoch(config,sess, ops, epoch=epoch)
            train_one_epoch(config,sess, ops, epoch=epoch)
            
            ACC_LOGGER.save(LOG_DIR)
            LOSS_LOGGER.save(LOG_DIR)
            ACC_LOGGER.plot(dest=LOG_DIR)
            LOSS_LOGGER.plot(dest=LOG_DIR)
            
            # Save the variables to disk.
            if epoch % config.save_period == 0 or epoch == end:
                checkpoint_path = os.path.join(config.log_dir, config.snapshot_prefix+str(epoch))
                saver.save(sess, checkpoint_path)
            

def train_one_epoch(config, sess, ops,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:config.num_points,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]
        num_batches = file_size // config.batch_size
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = (batch_idx+1) * config.batch_size
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            step, _, loss_val, pred_val = sess.run( [ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += config.batch_size
            loss_sum += loss_val
            if batch_idx % max(config.train_log_frq/config.batch_size,1) == 0:            
                acc = total_correct / float(total_seen)
                loss = loss_sum / float(num_batches)
                log_string('mean loss: %f' % loss)
                LOSS_LOGGER.log(loss, epoch, "train_loss")
                log_string('accuracy: %f' % acc)
                ACC_LOGGER.log(acc, epoch, "train_accuracy")
        
def eval_one_epoch(config, sess, ops, epoch=0):
    is_training = False
    num_votes = config.num_votes

    total_seen = 0
    loss_sum = 0
    predictions = []
    labels = []
    all = 0
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:config.num_points,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // config.batch_size + 1

        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = (batch_idx+1) * config.batch_size
            cur_batch_size = min(end_idx - start_idx, config.batch_size - end_idx + file_size)

            if cur_batch_size < config.batch_size:
                placeholder_data = np.zeros(([config.batch_size] + (list(current_data.shape))[1:]))
                placeholder_data[0:cur_batch_size, :, :] = current_data[start_idx:end_idx, :, :]
                
                placeholder_labels = np.zeros((config.batch_size))
                placeholder_labels[0:cur_batch_size] =  current_label[start_idx:end_idx]
                
                batch_labels = placeholder_labels
                batch_data = placeholder_data
            else:
                batch_data = current_data[start_idx:end_idx, :, :]
                batch_labels = current_label[start_idx:end_idx]
                
            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((config.batch_size, config.num_classes)) # score for classes
            
            for vote_idx in range(num_votes):
                rotated_data = provider.rotate_point_cloud_by_angle(batch_data, vote_idx/float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: batch_labels,
                             ops['is_training_pl']: is_training}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)

                batch_pred_sum += pred_val
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))

            pred_val = np.argmax(batch_pred_sum, 1)
            predictions += pred_val.tolist()[0:cur_batch_size]
            labels += current_label[start_idx:end_idx].tolist()
            
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum
                
                
    loss = loss_sum / float(total_seen)
    acc = sum([1 if predictions[i]==labels[i] else 0 for i in range(len(predictions))]) / float(len(predictions))
    print(loss)
    print(acc)
    
    if config.test:
        import Evaluation_tools as et
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir)  
    else:
        log_string('eval mean loss: %f' % loss)
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        log_string('eval accuracy: %f' % acc)
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
    
def test(config):     
    is_training = False
    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(config.batch_size, config.num_points)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config .allow_soft_placement = True
    tf_config.log_device_placement = True
    sess = tf.Session(config=tf_config)
    ld = config.log_dir
    ckptfile = os.path.join(ld,config.snapshot_prefix+str(config.weights))
    saver.restore(sess, ckptfile)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(config, sess, ops)
    
    
if __name__ == "__main__":
    if not config.test:
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        train(config)
        if config.weights == -1:
            config = add_to_config(config, 'weights', config.max_epoch)
        else:
            config = add_to_config(config, 'weights', config.max_epoch + config.weights)
        config = add_to_config(config, 'test', True)
    print(config)
    test(config)    
    LOG_FOUT.close()

