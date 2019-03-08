from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os
import random
import vgg19_trainable
import utils
from Logger import Logger
import skimage
from config import get_config

def read_lists(file, views=12):
    images = []
    labels = []
    with open(file, "r") as f:
        for line in f:
            line2 = line.split()[0]
            with open(line2, "r") as f2:
                cat = f2.readline().strip()
                f2.readline().strip()
                for view in range(views):
                    images.append(f2.readline().strip())
                    labels.append(int(cat))
    return images, labels

class Dataset():
    def __init__(self, files, labels, shuffle=False):
        self.files = files
        self.labels = labels
        self.shuffle = shuffle
        self.size = len(labels)
        self.index = 0
        self._reset()
        
    def next_batch(self, batch_size):
        reset = False
        if self.index >= self.size:
            self._reset()
            reset = True
            
        if self.index + batch_size >= self.size:
            files = self.files[self.index:]
            labels = self.labels[self.index:]
        else:
            files = self.files[self.index:self.index + batch_size]
            labels = self.labels[self.index:self.index + batch_size]
        images = [ skimage.io.imread(image,as_grey=True).reshape((224, 224, 1)) for image in files]
        self.index += batch_size
        
        return images, labels, reset
        
    def _reset(self):
        self.index = 0
        if self.shuffle:
            tmp = list(zip(self.files, self.labels))
            random.shuffle(tmp)
            self.files, self.labels = zip(*tmp)
      
def load_weights(config):
    if config.weights == -1:
        weights = config.pretrained_network_file
    else:
        weights = os.path.join(config.log_dir, config.snapshot_prefix + str(config.weights) + '.npy')
    return weights

def evaluate(test_data, config):
    print("Building net")
                       
    net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(config), trainable = True)
    with tf.device('/gpu:0'):
        sess = tf.Session()
        net.build()
        sess.run(tf.global_variables_initializer())
        _evaluate(net, sess, test_data, config)
        
def _evaluate(net, sess, test_data, config, epoch=0):
    print("starting evaluation")
    labels = []
    predictions = []
    losses = []
    acc = 0
    count = 0
    while True:
        batch_images, batch_labels, reset = test_data.next_batch(config.batch_size)
        if reset:
            break
        logits, loss = net.test(batch_images, batch_labels, sess)
        losses.append(loss)
        for i in range(len(batch_labels)/config.num_views):
            endindex = i*config.num_views + config.num_views
            prediction = np.argmax(np.sum(logits[i*config.num_views:endindex], axis=0))
            predictions.append(prediction)
            label = batch_labels[i*config.num_views]
            labels.append(label)
            if label == prediction:
                acc+=1
            else:
                acc+=0
            count+=1
    acc = acc/float(count)
    print("EVALUATING - acc: {} loss: {}".format(acc,loss))
    if not config.test:
        loss = np.mean(losses) 
        LOSS_LOGGER.log(loss, epoch, "eval_loss")
        ACC_LOGGER.log(acc, epoch, "eval_accuracy")
    else:
        import Evaluation_tools as et
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir)   
 
 
def train(train_data, test_data, config):
    print("Starting training")
    with tf.device('/gpu:0'):
        sess = tf.Session()
        
        start_epoch = 0
        WEIGHTS = config.weights
        if WEIGHTS!=-1:
            ld = config.log_dir
            start_epoch = WEIGHTS + 1
            ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),
                             os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = WEIGHTS)
            LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)),
                               os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = WEIGHTS)
        
        
        net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(config), trainable = True)    

        print("Weights loaded")
        net.build()
        sess.run(tf.global_variables_initializer())
        
        it_per_epoch =  train_data.size / config.batch_size
        for epoch in range(start_epoch, config.max_epoch+start_epoch +1):
            _evaluate(net, sess, test_data, config, epoch=epoch)
            accs = []
            losses = []
            for it in range(it_per_epoch):     
            
                batch_images, batch_labels, reset = train_data.next_batch(config.batch_size)
                _, loss, logits = net.train(batch_images, batch_labels, sess)
                acc = np.sum(np.argmax(logits, axis=1) == batch_labels)/float(len(batch_labels))
                accs.append(acc)
                losses.append(loss)
                
                if it%50 == 0:
                    loss = np.mean(losses)
                    acc = np.mean(accs)
                    print("TRAINING epoch: {} it: {}  loss: {} acc: {} ".format(epoch,it, loss, acc))
                    LOSS_LOGGER.log(loss, epoch, "train_loss")
                    ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    
                    ACC_LOGGER.save(config.log_dir)
                    LOSS_LOGGER.save(config.log_dir)
                    ACC_LOGGER.plot(dest=config.log_dir)
                    LOSS_LOGGER.plot(dest=config.log_dir)
                    
            if epoch%config.save_period == 0:
                net.save_npy(sess, os.path.join(config.log_dir, config.snapshot_prefix + str(epoch)))
           
            

def extract_features(config, train_data, test_data):
    print("Building net...")
    with tf.device('/gpu:0'):
        sess = tf.Session()
        net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(config), trainable = True)  
        net.build()
        sess.run(tf.global_variables_initializer())
        out_dir = config.data
        print("extracting test...")
        test_features, test_labels = _extract_features(test_data, net, sess, config)
        np.save(os.path.join(out_dir,'test_features.npy'), test_features)
        np.save(os.path.join(out_dir,'test_labels.npy'), test_labels)
        print("extracting train...")        
        train_features, train_labels = _extract_features(train_data, net, sess, config)        
        np.save(os.path.join(out_dir,'train_features.npy'), train_features)
        np.save(os.path.join(out_dir,'train_labels.npy'), train_labels)
          
def _extract_features(dataset, net, sess, config):
    
    per_batch = config.batch_size/config.num_views
    batch_size = per_batch * config.num_views
    features = np.zeros([dataset.size/config.num_views, config.num_views, 4096])
    labels = np.zeros([dataset.size/config.num_views], dtype = np.int32)
    it = 0
    while True:
        it += 1
        batch_images, labs, reset = dataset.next_batch(batch_size)
        if reset:
            break
        feat = net.extract(batch_images, sess)
        labs = labs[::config.num_views]
        offset = len(labs)
        feat = np.reshape(np.array(feat), (offset, config.num_views, 4096))
        index = per_batch*(it-1)
        features[index:index+offset] = feat
        offset = len(labs)
        labels[index:index+offset] = labs
        
    return features, labels
            
if __name__ == '__main__':
  
    config = get_config()
    if config.test:
        test_images, test_labels = read_lists(os.path.join(config.data,'test.txt'), views=config.num_views) 
        test_data = Dataset(test_images, test_labels, shuffle=False)
        evaluate(test_data, config)
    elif config.extract:
        test_images, test_labels = read_lists(os.path.join(config.data,'test.txt'), views=config.num_views) 
        test_data = Dataset(test_images, test_labels, shuffle=False)
        train_images, train_labels = read_lists(os.path.join(config.data,'train.txt'))
        train_data = Dataset(train_images, train_labels, shuffle=False)
        extract_features(config, train_data, test_data)
    else:
        test_images, test_labels = read_lists(os.path.join(config.data,'test.txt'), views=config.num_views) 
        test_data = Dataset(test_images, test_labels, shuffle=False)
        train_images, train_labels = read_lists(os.path.join(config.data,'train.txt'))
        train_data = Dataset(train_images, train_labels, shuffle=True)
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        train(train_data, test_data, config)
    
    