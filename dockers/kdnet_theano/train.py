from __future__ import print_function

import h5py as h5
import numpy as np
import os
from config import config
from kdnet import KDNET
from Logger import Logger
from lib.generators.meshgrid import generate_clouds
from lib.trees.kdtrees import KDTrees
from lib.nn.utils import dump_weights, load_weights

def iterate_minibatches(*arrays, **kwargs):
    if kwargs['mode'] == 'train':
        indices = np.random.choice((len(arrays[2]) - 1), 
                                   size=(len(arrays[2]) - 1)/kwargs['batchsize']*kwargs['batchsize'])
    elif kwargs['mode'] == 'test':
        indices = np.arange(len(arrays[2]) - 1)
    #indices = np.random.choice((len(arrays[2]) - 1), size=(len(arrays[2]) - 1)/kwargs['batchsize']*kwargs['batchsize'])
    if kwargs['mode']=='train' and kwargs['shuffle']:
        np.random.shuffle(indices)
        
    for start_idx in xrange(0, len(indices), kwargs['batchsize']):
        excerpt = indices[start_idx:start_idx + kwargs['batchsize']]
        tmp = generate_clouds(excerpt, kwargs['steps'], arrays[0], arrays[1], arrays[2])
        
        if kwargs['flip']:
            flip = np.random.random(size=(len(tmp), 2, 1))
            flip[flip >= 0.5] = 1.
            flip[flip < 0.5] = -1.
            tmp[:, :2] *= flip
        
        if kwargs['ascale']:
            tmp *= (kwargs['as_min'] + (kwargs['as_max'] - kwargs['as_min'])*np.random.random(size=(len(tmp), kwargs['dim'], 1)))
            tmp /= np.fabs(tmp).max(axis=(1, 2), keepdims=True)
        if kwargs['rotate']:
            r = np.sqrt((tmp[:, :2]**2).sum(axis=1))
            coss = tmp[:, 0]/r
            sins = tmp[:, 1]/r
            
            if kwargs['test_pos'] is not None:
                alpha = 2*np.pi*kwargs['test_pos']/kwargs['r_positions']
            else:
                alpha = 2*np.pi*np.random.randint(0, kwargs['r_positions'], (len(tmp), 1))/kwargs['positions']
                
            cosr = np.cos(alpha)
            sinr = np.sin(alpha)
            cos = coss*cosr - sins*sinr
            sin = sins*cosr + sinr*coss
            tmp[:, 0] = r*cos
            tmp[:, 1] = r*sin
            
        if kwargs['translate']:
            mins = tmp.min(axis=2, keepdims=True)
            maxs = tmp.max(axis=2, keepdims=True)
            rngs = maxs - mins
            tmp += kwargs['t_rate']*(np.random.random(size=(len(tmp), kwargs['dim'], 1)) - 0.5)*rngs
        
        trees_data = KDTrees(tmp, dim=kwargs['dim'], steps=kwargs['steps'], 
                             lim=kwargs['lim'], det=kwargs['det'], gamma=kwargs['gamma'])
            
        sortings, normals = trees_data['sortings'], trees_data['normals']
        if kwargs['input_features'] == 'all':
            clouds = np.empty((len(excerpt), kwargs['dim'], 2**kwargs['steps']), dtype=np.float32)
            for i, srt in enumerate(sortings):
                clouds[i] = tmp[i, :, srt].T
        elif kwargs['input_features'] == 'no':
            clouds = np.ones((len(excerpt), 1, 2**kwargs['steps']), dtype=np.float32)
        
        if kwargs['mode'] == 'train':
            yield [clouds] + normals[::-1] + [arrays[3][excerpt]]
        if kwargs['mode'] == 'test':
            yield [clouds] + normals[::-1] + [arrays[3][excerpt]]


def get_probs(net, vertices, faces, nFaces, labels, **kwargs):
    prob_sum = np.zeros((len(nFaces)-1, kwargs['n_output']), dtype=np.float32)
    losses = []
    accuracies = []
    for ens in xrange(kwargs['n_ens']):
        probability = np.zeros((len(nFaces)-1, kwargs['n_output']), dtype=np.float32)
        index = 0    
        for i, batch in enumerate(iterate_minibatches(vertices, faces, nFaces,labels, **kwargs)):
            loss, probs, acc = net.prob_fun(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11])
            losses.append(loss)
            accuracies.append(acc)
            size_of_batch = batch[-1].shape[0]
            probability[index:index+size_of_batch] += probs
            index += size_of_batch
            #probability[batch[-1]] += probs
        prob_sum += probability

    return np.mean(losses), prob_sum / kwargs['n_ens'], np.mean(accuracies)


def acc_fun(net, vertices, faces, nFaces, labels, **kwargs):
    loss, probs, acc = get_probs(net, vertices, faces, nFaces, labels, **kwargs)
    return loss, probs.argmax(axis=1), acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/converted", type=str, help="Path to the dataset")
    parser.add_argument("--log_dir", default="logs", type=str, help="logging directory")
    parser.add_argument("--max_epoch", type = int, default=100, help="Number of epochs to train for")
    parser.add_argument("--save_each", type = int, default=5, help="How often to save the model")
    parser.add_argument('--weights',default=None, type=int, help='Number of model to finetune or evaluate')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    print(config)
    
    print("Reading data...")
    path2data = os.path.join(args.data, 'data.h5')
    with h5.File(path2data, 'r') as hf:
        if not args.test:
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
    
    if args.weights!=None:
        weights = args.weights
        load_weights(os.path.join(args.log_dir, 'model-{}.pkl'.format(weights)), net.KDNet['output'])
        print("Loaded weights")
    
    if args.test:
        config['mode'] = 'test'
        _, predictions = acc_fun(test_vertices, test_faces, test_nFaces, test_labels, **config) 
        acc = 100.*(predictions == test_labels).sum()/len(test_labels)
        
        print('Eval accuracy:  {}'.format(acc))
        import Evaluation_tools as et
        eval_file = os.path.join(args.log_dir, 'kdnet.txt')
        et.write_eval_file(args.data, eval_file, predictions , test_labels , 'KDNET')
        et.make_matrix(args.data, eval_file, args.log_dir)
        
    else:
        print("Starting training")
        LOSS_LOGGER = Logger("kdnet_loss")
        ACC_LOGGER = Logger("kdnet_acc")
        start_epoch = 0
        if args.weights!=None:
            start_epoch = weights
            ACC_LOGGER.load((os.path.join(args.log_dir,"kdnet_acc_train_accuracy.csv"),os.path.join(args.log_dir,"kdnet_acc_eval_accuracy.csv")), epoch=weights)
            LOSS_LOGGER.load((os.path.join(args.log_dir,"kdnet_loss_train_loss.csv"), os.path.join(args.log_dir,'kdnet_loss_eval_loss.csv')), epoch=weights)
        num_epochs = args.max_epoch
        num_save = args.save_each
        for epoch in xrange(start_epoch, num_epochs+start_epoch):
            
            config['mode'] = 'test'
            loss, predictions = acc_fun(net,test_vertices, test_faces, test_nFaces, test_labels, **config)
            acc = (predictions == test_labels).sum()/float(len(test_labels))
            print("evaluating loss:{} acc:{}".format(loss,acc))       
            LOSS_LOGGER.log(loss, epoch, "eval_loss")
            ACC_LOGGER.log(acc, epoch, "eval_accuracy")
            
            config['mode'] = 'train'
            losses = []
            accuracies = []
            for i, batch in enumerate(iterate_minibatches(train_vertices, train_faces, train_nFaces, train_labels, **config)):
                train_err_batch, train_acc_batch = net.train_fun(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11])
                
                losses.append(train_err_batch)
                accuracies.append(train_acc_batch)

                if i % 20 == 0:
                    loss = np.mean(losses)
                    acc = np.mean(accuracies)
                    LOSS_LOGGER.log(loss, epoch, "train_loss")
                    ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    print('EPOCH {}, batch {}: loss {} acc {}'.format(epoch, i, loss, acc))
                    losses = []
                    accuracies = []
            
                          

            

            
            ACC_LOGGER.save(args.log_dir)
            LOSS_LOGGER.save(args.log_dir)
            ACC_LOGGER.plot(dest=args.log_dir)
            LOSS_LOGGER.plot(dest=args.log_dir)
            if epoch % num_save == 0:
                dump_weights(os.path.join(args.log_dir, 'model-{}.pkl'.format(epoch)), net.KDNet['output'])
                
                