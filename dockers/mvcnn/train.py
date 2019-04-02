from __future__ import print_function
import numpy as np
import os,sys,inspect
import tensorflow as tf
import os
import math
import hickle as hkl
import sklearn.metrics as metrics

from input import Dataset
from Logger import Logger
from config import get_config, add_to_config

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

def train(dataset_train, dataset_test, caffemodel=''):
    print ('train() called')
    V = config.num_views
    batch_size =config.batch_size
    
    dataset_train.shuffle()
    data_size = dataset_train.size()
    
    print ('training size:', data_size)

    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            
            tf_config=tf.ConfigProto(log_device_placement=False)
            tf_config.gpu_options.allow_growth = True
            tf_config.allow_soft_placement = True
                                 
            global_step = tf.Variable(0, trainable=False)
             
            # placeholders for graph input
            view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
            y_ = tf.placeholder('int64', shape=(None), name='y')
            keep_prob_ = tf.placeholder('float32')
            
            # graph outputs
            fc8 = model.inference_multiview(view_, config.num_classes, keep_prob_)
            loss = model.loss(fc8, y_)
            train_op = model.train(loss, global_step, data_size)
            prediction = model.classify(fc8)    
            placeholders = [view_, y_, keep_prob_, prediction, loss]
            validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
            validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')

            saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)
    
            init_op = tf.global_variables_initializer()
            sess = tf.Session(config=tf_config)
            weights = config.weights
            if weights==-1:
                startepoch = 0
                if caffemodel:
                    sess.run(init_op)
                    model.load_alexnet_to_mvcnn(sess, caffemodel)
                    print ('loaded pretrained caffemodel:', caffemodel)
                else:
                    sess.run(init_op)
                    print ('init_op done')  
            else:
                ld = config.log_dir
                startepoch = weights + 1
                ckptfile = os.path.join(ld,config.snapshot_prefix+str(weights))

                saver.restore(sess, ckptfile)
                print ('restore variables done')
    
            total_seen = 0
            total_correct = 0
            total_loss = 0
            
            step = 0
            begin = startepoch
            end = config.max_epoch + startepoch
            for epoch in xrange(begin, end + 1):
                acc, eval_loss, predictions, labels = _test(dataset_test, config, sess, placeholders)
                print ('epoch %d: step %d, validation loss=%.4f, acc=%f' % (epoch, step, eval_loss, acc*100.))
                
                LOSS_LOGGER.log(eval_loss, epoch, "eval_loss")
                ACC_LOGGER.log(acc, epoch, "eval_accuracy")
                ACC_LOGGER.save(config.log_dir)
                LOSS_LOGGER.save(config.log_dir)
                ACC_LOGGER.plot(dest=config.log_dir)
                LOSS_LOGGER.plot(dest=config.log_dir)
                
                for batch_x, batch_y in dataset_train.batches(batch_size):
                    step += 1
    
                    feed_dict = {view_: batch_x,
                                 y_ : batch_y,
                                 keep_prob_: 0.5 }
    
                    _, pred, loss_value = sess.run(
                            [train_op, prediction,  loss,],
                            feed_dict=feed_dict)
                    
                    total_loss += loss_value
                    correct = np.sum(pred == batch_y)
                    total_correct+=correct
                    total_seen+=batch_size
                    
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    
                    if step % max(config.train_log_frq/config.batch_size,1) == 0:
                        acc_ = total_correct / float(total_seen)
                        ACC_LOGGER.log(acc_, epoch, "train_accuracy")
                        loss_ = total_loss / float(total_seen/batch_size) 
                        LOSS_LOGGER.log(loss_, epoch,"train_loss")           
                        print ('epoch %d step %d, loss=%.2f, acc=%.2f' %(epoch, step, loss_, acc_))
                        total_seen = 0
                        total_correct = 0
                        total_loss = 0
                        
                if epoch % config.save_period == 0 or epoch == end:
                    checkpoint_path = os.path.join(config.log_dir, config.snapshot_prefix+str(epoch))
                    saver.save(sess, checkpoint_path)
                            

def test(dataset, config):
    print ('test() called')
    weights = config.weights
    V = config.num_views
    batch_size = config.batch_size
    ckptfile = os.path.join(config.log_dir,config.snapshot_prefix+str(weights))
    data_size = dataset.size()
    print ('dataset size:', data_size)

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        
        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')
        
        fc8 = model.inference_multiview(view_, config.num_classes, keep_prob_)
        loss = model.loss(fc8, y_)
        #train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)
        placeholders = [view_, y_, keep_prob_, prediction, loss]
        saver = tf.train.Saver(tf.all_variables())

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        saver.restore(sess, ckptfile)
        print ('restore variables done')
        print ("Start testing")
        print ("Size:", data_size)
        print ("It'll take", int(math.ceil(data_size/batch_size)), "iterations.")

        acc, _, predictions, labels = _test(dataset, config, sess, placeholders)
        print ('acc:', acc*100)
    
    import Evaluation_tools as et
    eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
    et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
    et.make_matrix(config.data, eval_file, config.log_dir)    

def _test(dataset, config, sess, placeholders):
    val_losses = []
    predictions = []
    labels = []
    for batch_x, batch_y in dataset.batches(config.batch_size):

        feed_dict = {placeholders[0] : batch_x,
                     placeholders[1] : batch_y,
                     placeholders[2] : 1.0}

        pred, loss_value = sess.run(
                [placeholders[3],  placeholders[4],],
                feed_dict=feed_dict)
        val_losses.append(loss_value)
        predictions.extend(pred.tolist())
        labels.extend(batch_y.tolist())
    loss = np.mean(val_losses)
    acc = metrics.accuracy_score(labels, predictions) 
    return acc, loss, predictions, labels


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels
    
if __name__ == '__main__':
    config = get_config()
    
    data_path = config.data
    test = 0
    with open(os.path.join(data_path, 'test.txt'), 'r') as f:
        for line in f:
            if os.path.exists(line.split()[0]):
                pass
            else:
                test+=1
                print(line)
   
    print ('start loading data')
    
    listfiles_test, labels_test = read_lists(os.path.join(data_path, 'test.txt'))
    dataset_test = Dataset(listfiles_test, labels_test, subtract_mean=False, V=config.num_views)
        
    if not config.test:
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        listfiles_train, labels_train = read_lists(os.path.join(data_path, 'train.txt'))
        dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=config.num_views)       
        train(dataset_train, dataset_test, config.pretrained_network_file)
        
        if config.weights == -1:
            config = add_to_config(config, 'weights', config.max_epoch)
        else:
            config = add_to_config(config, 'weights', config.max_epoch + config.weights)
        config = add_to_config(config, 'test', True)  
            
    test(dataset_test, config)
