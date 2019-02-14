from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os
import random
from vgg import vgg19_trainable
from vgg import utils
from Logger import Logger
import skimage

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
      
def load_weights(args):
    if args.weights == -1:
        weights = 'vgg/vgg19.npy'
    else:
        weights= os.path.join(args.log_dir,'./vgg_tuned_{}.npy'.format(args.weights))
    return weights

def evaluate(test_data, args):
    print("Building net")
                       
    net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(args), trainable = True)
    with tf.device('/cpu:0'):
        sess = tf.Session()
        net.build()
        sess.run(tf.global_variables_initializer())
        _evaluate(net,sess, test_data, args)
        
def _evaluate(net, sess, test_data, args):
    print("starting evaluation")
    losses = []
    #labels = []
    predictions = []
    accs = []
    iter = 0
    while True:
        batch_images, batch_labels, reset = test_data.next_batch(args.batch_size)
        if reset:
            break
        if not args.test and iter>10:
            break
        logits, loss = net.test(batch_images, batch_labels, sess)
        losses.append(loss)
        preds = np.argmax(logits, axis=1)
        predictions += list(preds)
        #labels += batch_labels
        acc = np.sum(preds == batch_labels)/float(len(batch_labels))
        accs.append(acc)
        
        acc = np.mean(accs)
        loss = np.mean(losses)

        print("EVALUATING - loss: {} acc: {} ".format(loss, acc))
    if not args.test:
        loss = np.mean(losses) 
        acc = np.mean(accs)
        LOSS_LOGGER.log(loss, 0, "eval_loss")
        ACC_LOGGER.log(acc, 0, "eval_accuracy")
    else:
        import Evaluation_tools as et
        eval_file = os.path.join(args.log_dir, 'vgg.txt')
        et.write_eval_file(args.data, eval_file, predictions , test_labels , 'VGG')
        et.make_matrix(args.data, eval_file, args.log_dir)
 
def train(train_data, test_data, args):
   
    print("Starting training")
    with tf.device('/cpu:0'):
        sess = tf.Session()
        if args.weights == -1:
            it = 0
        else:
            it = args.weights + 1
            ACC_LOGGER.load((os.path.join(args.log_dir,"vgg_acc_train_accuracy.csv"),os.path.join(args.log_dir,"vgg_acc_eval_accuracy.csv")), epoch=weights)
            LOSS_LOGGER.load((os.path.join(args.log_dir,"vgg_loss_train_loss.csv"), os.path.join(args.log_dir,'vgg_loss_eval_loss.csv')), epoch=weights)
        
        net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(args), trainable = True)    

        print("Weights loaded")
        net.build()
        sess.run(tf.global_variables_initializer())
        
        accs = []
        losses = []
        while True:
            
            if it%50 == 0:
                net.save_npy(sess, os.path.join(args.log_dir,'./vgg_tuned_{}.npy'.format(it)))
                _evaluate(net, sess, test_data, args)
            if it == args.max_iter:
                break
            
            
            batch_images, batch_labels, reset = train_data.next_batch(args.batch_size)
            _, loss, logits = net.train(batch_images, batch_labels, sess)
            acc = np.sum(np.argmax(logits, axis=1) == batch_labels)/float(len(batch_labels))
            accs.append(acc)
            losses.append(loss)
            
            print("TRAINING it: {}  loss: {} acc: {} ".format(it, loss, acc))
            
            if it%10 == 0:
                LOSS_LOGGER.log(np.mean(losses), 0, "train_loss")
                ACC_LOGGER.log(np.mean(accs), 0, "train_accuracy")
                
                ACC_LOGGER.save(args.log_dir)
                LOSS_LOGGER.save(args.log_dir)
                ACC_LOGGER.plot(dest=args.log_dir)
                LOSS_LOGGER.plot(dest=args.log_dir)
            

            it += 1

def extract_features(args, train_data, test_data):
    print("Building net...")
    net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(args), trainable = True)   
    
    with tf.device('/cpu:0'):
        sess = tf.Session()
        net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(args), trainable = True)  
        net.build()
        sess.run(tf.global_variables_initializer())
        out_dir = args.data
        
        test_features, test_labels = _extract_features(test_data, net, sess, args)
        np.save(os.path.join(out_dir,'test_features.npy'), test_features)
        np.save(os.path.join(out_dir,'test_labels.npy'), test_labels)
        
        train_features, train_labels = _extract_features(train_data, net, sess, args)        
        np.save(os.path.join(out_dir,'train_features.npy'), train_features)
        np.save(os.path.join(out_dir,'train_labels.npy'), train_labels)
          
def _extract_features(dataset, net, sess, args):
    
    per_batch = args.batch_size/args.views
    batch_size = per_batch * args.views
    print(batch_size)
    features = np.zeros([dataset.size/args.views, args.views, 4096])
    labels = np.zeros([dataset.size/args.views], dtype = np.int32)
    it = 0
    while True:
        it += 1
        batch_images, labs, reset = dataset.next_batch(batch_size)
        
        if reset:
            break
        feat = net.extract(batch_images, sess)
        labs = labs[::args.views]
        offset = len(labs)
        feat = np.reshape(np.array(feat), (offset, args.views, 4096))
        index = per_batch*(it-1)
        print(index, offset)
        features[index:index+offset] = feat
        offset = len(labs)
        labels[index:index+offset] = labs
        
    return features, labels
            
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/data/converted', type=str)
    parser.add_argument('--batch_size',type=int, default=200)
    parser.add_argument('--num_cats',type=int, default=40)
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--extract', action = 'store_true')    
    parser.add_argument('--weights', default=-1, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--views', default=12, type=int)
    parser.add_argument('--max_iter', default=2000, type=int)    
    
    args = parser.parse_args()

    if args.test:
        test_images, test_labels = read_lists(os.path.join(args.data,'test.txt'), views=args.views) 
        test_data = Dataset(test_images, test_labels, shuffle=False)
        evaluate(test_data, args)
    elif args.extract:
        test_images, test_labels = read_lists(os.path.join(args.data,'test.txt'), views=args.views) 
        test_data = Dataset(test_images, test_labels, shuffle=False)
        train_images, train_labels = read_lists(os.path.join(args.data,'train.txt'))
        train_data = Dataset(train_images, train_labels, shuffle=False)
        extract_features(args, train_data, test_data)
    else:
        test_images, test_labels = read_lists(os.path.join(args.data,'test.txt'), views=args.views) 
        test_data = Dataset(test_images, test_labels, shuffle=False)
        train_images, train_labels = read_lists(os.path.join(args.data,'train.txt'))
        train_data = Dataset(train_images, train_labels, shuffle=True)
        LOSS_LOGGER = Logger("vgg_loss")
        ACC_LOGGER = Logger("vgg_acc")
        train(train_data, test_data, args)
    
    