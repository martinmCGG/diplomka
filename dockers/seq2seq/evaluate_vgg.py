from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os
import random
from vgg import vgg19_trainable
from vgg import utils
from train_vgg import Dataset
from train_vgg import load_weights
from train_vgg import read_lists

def evaluate(test_data, args):
    print("Building net")
                       
    net = vgg19_trainable.Vgg19(vgg19_npy_path=load_weights(args), trainable = True)
    with tf.device('/cpu:0'):
        sess = tf.Session()
        net.build()
        sess.run(tf.global_variables_initializer())

        print("starting evaluation")

        labels = []
        predictions = []
        accs = []

        while True:
            batch_images, batch_labels, reset = test_data.next_batch(args.views)
            if reset:
                break

            logits, _ = net.test(batch_images, batch_labels, sess)
            prediction = np.argmax(np.sum(logits, axis=0))
            
            predictions.append(prediction)
            label = batch_labels[0]
            labels.append(label)
            
            if label == prediction:
                accs.append(1)
            else:
                accs.append(0)
            
            acc = np.mean(accs)
  
            print("EVALUATING - acc: {} ".format(acc))

    import Evaluation_tools as et
    eval_file = os.path.join(args.log_dir, 'vgg.txt')
    et.write_eval_file(args.data, eval_file, predictions , labels , 'VGG')
    et.make_matrix(args.data, eval_file, args.log_dir)

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/data/converted', type=str)
    parser.add_argument('--batch_size',type=int, default=200)
    parser.add_argument('--num_cats',type=int, default=40)
    parser.add_argument('--weights', default=-1, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--views', default=12, type=int)
  
    
    args = parser.parse_args()
    
    test_images, test_labels = read_lists(os.path.join(args.data,'test.txt'), views=args.views) 
    test_data = Dataset(test_images, test_labels, shuffle=False)
    evaluate(test_data, args)