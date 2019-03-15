from __future__ import print_function

import h5py as h5
import numpy as np
import os
from kdnet import KDNET
from Logger import Logger
from lib.generators.meshgrid import generate_clouds
from lib.trees.kdtrees import KDTrees
from lib.nn.utils import dump_weights, load_weights
from config import get_config

def iterate_minibatches(*arrays, **kwargs):
    mode = kwargs['mode']
    config = kwargs['config']
    if mode == 'train':
        indices = np.random.choice((len(arrays[2]) - 1), 
                                   size=(len(arrays[2]) - 1)/config.batch_size*config.batch_size)
    elif mode == 'test':
        indices = np.arange(len(arrays[2]) - 1)
    #indices = np.random.choice((len(arrays[2]) - 1), size=(len(arrays[2]) - 1)/config.batch_size*config.batch_size)
    if mode=='train' and config.shuffle:
        np.random.shuffle(indices)
        
    for start_idx in xrange(0, len(indices), config.batch_size):
        excerpt = indices[start_idx:start_idx + config.batch_size]
        tmp = generate_clouds(excerpt, config.steps, arrays[0], arrays[1], arrays[2])
        
        if config.flip:
            flip = np.random.random(size=(len(tmp), 2, 1))
            flip[flip >= 0.5] = 1.
            flip[flip < 0.5] = -1.
            tmp[:, :2] *= flip
        
        if config.ascale:
            tmp *= (config.as_min + config.as_max - config.as_min)*np.random.random(size=(len(tmp), config.dim, 1))
            tmp /= np.fabs(tmp).max(axis=(1, 2), keepdims=True)
        if config.rotate:
            r = np.sqrt((tmp[:, :2]**2).sum(axis=1))
            coss = tmp[:, 0]/r
            sins = tmp[:, 1]/r
            
            if config != -1:
                alpha = 2*np.pi*config.test_pos/config.r_positions
            else:
                alpha = 2*np.pi*np.random.randint(0, config.r_positions, (len(tmp), 1))/config.r_positions
                
            cosr = np.cos(alpha)
            sinr = np.sin(alpha)
            cos = coss*cosr - sins*sinr
            sin = sins*cosr + sinr*coss
            tmp[:, 0] = r*cos
            tmp[:, 1] = r*sin
            
        if config.translate:
            mins = tmp.min(axis=2, keepdims=True)
            maxs = tmp.max(axis=2, keepdims=True)
            rngs = maxs - mins
            tmp += config.t_rate*(np.random.random(size=(len(tmp), config.dim, 1)) - 0.5)*rngs
        
        trees_data = KDTrees(tmp, dim=config.dim, steps=config.steps, 
                             lim=config.lim, det=config.det, gamma=config.gamma)
            
        sortings, normals = trees_data['sortings'], trees_data['normals']
        if config.input_features == 'all':
            clouds = np.empty((len(excerpt), config.dim, 2**config.steps), dtype=np.float32)
            for i, srt in enumerate(sortings):
                clouds[i] = tmp[i, :, srt].T
        elif config.input_features == 'no':
            clouds = np.ones((len(excerpt), 1, 2**config.steps), dtype=np.float32)
        
        if mode == 'train':
            yield [clouds] + normals[::-1] + [arrays[3][excerpt]]
        if mode == 'test':
            yield [clouds] + normals[::-1] + [arrays[3][excerpt]]


def get_probs(net, vertices, faces, nFaces, labels, **kwargs):
    mode = kwargs['mode']
    config = kwargs['config']
    prob_sum = np.zeros((len(nFaces)-1, config.num_classes), dtype=np.float32)
    losses = []
    for ens in xrange(config.num_votes):
        probability = np.zeros((len(nFaces)-1, config.num_classes), dtype=np.float32)
        index = 0    
        for i, batch in enumerate(iterate_minibatches(vertices, faces, nFaces,labels, config=config, mode=mode)):
            loss, probs = net.prob_fun(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11])
            losses.append(loss)
            size_of_batch = batch[-1].shape[0]
            probability[index:index+size_of_batch] += probs
            index += size_of_batch
        prob_sum += probability
    return np.mean(losses), prob_sum / config.num_votes

def acc_fun(net, vertices, faces, nFaces, labels, **kwargs):
    mode = kwargs['mode']
    config = kwargs['config']
    loss, probs= get_probs(net, vertices, faces, nFaces, labels, config=config, mode=mode)
    return loss, probs.argmax(axis=1)

if __name__ == "__main__":

    config = get_config()
    
    print("Reading data...")
    path2data = os.path.join(config.data, 'data.h5')
    with h5.File(path2data, 'r') as hf:
        if not config.test:
            train_vertices = np.array(hf.get('train_vertices'))
            train_faces = np.array(hf.get('train_faces'))
            train_nFaces = np.array(hf.get('train_nFaces'))
            train_labels = np.array(hf.get('train_labels'))
        test_vertices = np.array(hf.get('test_vertices'))
        test_faces = np.array(hf.get('test_faces'))
        test_nFaces = np.array(hf.get('test_nFaces'))
        test_labels = np.array(hf.get('test_labels'))
            
    print("Compiling net...")
    net = KDNET(config)    
    
    if config.weights != -1:
        weights = config.weights
        load_weights(os.path.join(config.log_dir, config.snapshot_prefix+str(weights)), net.KDNet['output'])
        print("Loaded weights")
    
    if config.test:
        print("Start testing")
        _, predictions = acc_fun(net,test_vertices, test_faces, test_nFaces, test_labels, mode='test',config=config) 
        acc = 100.*(predictions == test_labels).sum()/len(test_labels)
        
        print('Eval accuracy:  {}'.format(acc))
        import Evaluation_tools as et
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, test_labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir)  
    else:
        print("Start training")
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        start_epoch = 0
        if config.weights != -1:
            ld = config.log_dir
            WEIGHTS = config.weights
            ckptfile = os.path.join(ld,config.snapshot_prefix+str(WEIGHTS))
            start_epoch = WEIGHTS + 1
            ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = WEIGHTS)
            LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)), os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = WEIGHTS)
          
        for epoch in xrange(start_epoch,config.max_epoch+start_epoch):
            
            loss, predictions = acc_fun(net,test_vertices, test_faces, test_nFaces, test_labels, mode='test',config=config)
            acc = (predictions == test_labels).sum()/float(len(test_labels))
            print("evaluating loss:{} acc:{}".format(loss,acc))       
            LOSS_LOGGER.log(loss, epoch, "eval_loss")
            ACC_LOGGER.log(acc, epoch, "eval_accuracy")
            
            losses = []
            accuracies = []
            for i, batch in enumerate(iterate_minibatches(train_vertices, train_faces, train_nFaces, train_labels, mode='train', config=config)):
                train_err_batch, train_acc_batch = net.train_fun(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11])
                
                losses.append(train_err_batch)
                accuracies.append(train_acc_batch)

                if i % max(config.train_log_frq/config.batch_size,1) == 0:
                    loss = np.mean(losses)
                    acc = np.mean(accuracies)
                    LOSS_LOGGER.log(loss, epoch, "train_loss")
                    ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    print('EPOCH {}, batch {}: loss {} acc {}'.format(epoch, i, loss, acc))
                    losses = []
                    accuracies = []
            
            ACC_LOGGER.save(config.log_dir)
            LOSS_LOGGER.save(config.log_dir)
            ACC_LOGGER.plot(dest=config.log_dir)
            LOSS_LOGGER.plot(dest=config.log_dir)
            if epoch % config.save_period == 0:
                dump_weights(os.path.join(config.log_dir, config.snapshot_prefix+str(epoch)), net.KDNet['output'])
                
                