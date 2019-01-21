import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
import hickle as hkl
import os.path as osp
from glob import glob
import sklearn.metrics as metrics

from input import Dataset
import globals as g_
from Logger import Logger


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

FLAGS = tf.app.flags.FLAGS
try:
    os.mkdir('./logs/')
except:
    pass


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', osp.dirname(sys.argv[0]) + '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '', 
                            """finetune with a pretrained model""")
tf.app.flags.DEFINE_string('caffemodel', '', 
                            """finetune with a model converted by caffe-tensorflow""")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='logs', help='Directory where to write event logs [default: log]')
parser.add_argument('--log_device_placement', default=False, help='Whether to log device placement.' )
parser.add_argument('--weights',type=int, help='Number of model weights')
parser.add_argument('--caffemodel', default='alexnet_imagenet.npy', help='Directory where to write event logs [default: log]')

args = parser.parse_args()
args.log_dir = args.train_dir
np.set_printoptions(precision=3)


def train(dataset_train, dataset_val, weights='', caffemodel=''):
    print 'train() called'
    V = g_.NUM_VIEWS
    batch_size = g_.BATCH_SIZE
    
    dataset_train.shuffle()
    dataset_val.shuffle()
    data_size = dataset_train.size()
    print 'training size:', data_size

    with tf.Graph().as_default():

        if not bool(weights):
            startepoch = 0
        else:
            startepoch = weights + 1
            ckptfile = os.path.join(args.train_dir, 'model.ckpt-'+str(weights))
            ACC_LOGGER.load((os.path.join(args.log_dir,"mvcnn_acc_train_accuracy.csv"),os.path.join(args.log_dir,"mvcnn_acc_eval_accuracy.csv")), epoch = weights)
            LOSS_LOGGER.load((os.path.join(args.log_dir,"mvcnn_loss_train_loss.csv"), os.path.join(args.log_dir,'mvcnn_loss_eval_loss.csv')), epoch = weights)                            
        global_step = tf.Variable(0, trainable=False)
         
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
        loss = model.loss(fc8, y_)
        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)

        # build the summary operation based on the F colection of Summaries
        summary_op = tf.summary.merge_all()


        # must be after merge_all_summaries
        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
        validation_summary = tf.summary.scalar('validation_loss', validation_loss)
        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
        validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_acc)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=args.log_device_placement))
        
        if args.weights:
            # load checkpoint file
            saver.restore(sess, ckptfile)
            print 'restore variables done'
        elif caffemodel:
            # load caffemodel generated with caffe-tensorflow
            sess.run(init_op)
            model.load_alexnet_to_mvcnn(sess, caffemodel)
            print 'loaded pretrained caffemodel:', caffemodel
        else:
            # from scratch
            sess.run(init_op)
            print 'init_op done'

        summary_writer = tf.summary.FileWriter(args.train_dir,
                                               graph=sess.graph) 

        total_seen = 0
        total_correct = 0
        total_loss = 0
        
        step = 0
        for epoch in xrange(startepoch, g_.TRAIN_FOR + startepoch):
            print 'epoch:', epoch
            
            for batch_x, batch_y in dataset_train.batches(batch_size):
                step += 1

                start_time = time.time()
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
                
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                log_period = 10
                if step % log_period == 0:
                    sec_per_batch = float(duration)
                    print '%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                         % (datetime.now(), step, loss_value,
                                    FLAGS.batch_size/duration, sec_per_batch)
                    
                    acc = total_correct / float(total_seen)
                    ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    loss_ = total_loss / float(log_period) 
                    LOSS_LOGGER.log(loss_, epoch,"train_loss")                   
                    
                    total_seen = 0
                    total_correct = 0
                    total_loss = 0
                    
                if step % 100 == 0:
                    # print 'running summary'
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

            if epoch % g_.SAVE_PERIOD == 0 and epoch>0:
                checkpoint_path = os.path.join(args.train_dir, 'model.ckpt-'+str(epoch))
                saver.save(sess, checkpoint_path)
                    
            val_losses = []
            predictions = []
            labels = []
                
            for batch_x, batch_y in dataset_val.batches(batch_size):
    
                start_time = time.time()
                feed_dict = {view_: batch_x,
                             y_ : batch_y,
                             keep_prob_: 1.0}
    
                pred, loss_value = sess.run(
                        [prediction,  loss,],
                        feed_dict=feed_dict)
                
                val_losses.append(loss_value)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    
                predictions.extend(pred.tolist())
                labels.extend(batch_y.tolist())
            val_loss = np.mean(val_losses)
            acc = metrics.accuracy_score(labels, predictions)     
            print '%s: step %d, validation loss=%.4f, acc=%f' %\
                    (datetime.now(), step, val_loss, acc*100.)
            LOSS_LOGGER.log(val_loss, epoch, "eval_loss")
            ACC_LOGGER.log(acc, epoch, "eval_accuracy")
            
            ACC_LOGGER.save(args.train_dir)
            LOSS_LOGGER.save(args.train_dir)
            ACC_LOGGER.plot(dest=args.train_dir)
            LOSS_LOGGER.plot(dest=args.train_dir)
            # validation summary
            val_loss_summ = sess.run(validation_summary,
                    feed_dict={validation_loss: val_loss})
            val_acc_summ = sess.run(validation_acc_summary, 
                    feed_dict={validation_acc: acc})
            summary_writer.add_summary(val_loss_summ, step)
            summary_writer.add_summary(val_acc_summ, step)
            summary_writer.flush()
    
    
def main(argv):
    st = time.time() 
    print 'start loading data'

    listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    listfiles_val, labels_val = read_lists(g_.VAL_LOL)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=g_.NUM_VIEWS)

    print 'done loading data, time=', time.time() - st

    train(dataset_train, dataset_val, args.weights, args.caffemodel)


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels
    
    
if __name__ == '__main__':
    LOSS_LOGGER = Logger("mvcnn_loss")
    ACC_LOGGER = Logger("mvcnn_acc")
    main(sys.argv)


