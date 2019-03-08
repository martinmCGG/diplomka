import tensorflow as tf
import numpy as np
import os
from utils import data_utils
import random
from config import get_config
from tensorflow.contrib.learn.python.learn.datasets import base

config = get_config()

class DataSet(object):
    def __init__(self, filenames, fcs, labels):
        self._filenames, self._fcs, self._labels = filenames, fcs, labels
        self._epochs_completed, self._index_in_epoch = 0, 0
        self._num_examples = fcs.shape[0]

    @property
    def labels(self):
        return self._labels
    @property
    def fcs(self):
        return self._fcs

    def size(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True, as_sequence=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._fcs = self._fcs[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            fcs_rest_part = self._fcs[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._fcs = self._fcs[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            fcs_new_part = self._fcs[start:end]
            labels_new_part = self._labels[start:end]
            if as_sequence:
                return np.concatenate((fcs_rest_part, fcs_new_part), axis=0), self.batch_label2sequence(np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0))
            else:
                return np.concatenate((fcs_rest_part, fcs_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            if as_sequence:
                return self._fcs[start:end], self.batch_label2sequence(self._labels[start:end])
            else:
                return self._fcs[start:end], self._labels[start:end]

    def label2sequence(self, label_onehot):
        label = np.argmax(label_onehot) + 1
        sequence = [data_utils.GO_ID]
        for i in range(1, np.shape(label_onehot)[0]+1):
            if label != i:
                sequence.append(2*i)
            else:
                sequence.append(2*i-1)
        return np.array(sequence)

    def batch_label2sequence(self, labels_onehot):
        return np.array([self.label2sequence(label_onehot) for label_onehot in labels_onehot])


def read_data(data_dir, config, roll_number=12, read_train=True, read_test=True):
    print("read data from %s" %data_dir)
    train_dataset, test_dataset = None, None
    if read_train:
        train_fcs = np.load(os.path.join(data_dir, config.train_feature_file))
        #train_fcs = np.array([[[1,1,1,1],[11,11,11,11]],[[2,2,2,2],[22,22,22,22]],[[3,3,3,3],[33,33,33,33]],[[4,4,4,4],[44,44,44,44]],[[5,5,5,5],[55,55,55,55]], [[6,6,6,6],[66,66,66,66]]])
        train_fcs = multiview(train_fcs, config.num_views)
        train_labels = np.load(os.path.join(config.data, config.train_label_file))
        #train_labels = np.array([0,1,1,2,3,4])
        if config.enrich_shapenet:
            train_labels_range = labels_statistic(train_labels)
        train_labels = onehot(train_labels)
        if config.enrich_data:
            train_fcs, train_labels = roll_enrich(train_fcs, train_labels, roll_number)
            print(train_fcs.shape, train_labels.shape)
        if config.enrich_shapenet:
            for label, statistic in train_labels_range.items():
                start, count = statistic[0], statistic[1]
                if count < config.min_shapenet_class_count:
                    repeat_num = config.min_shapenet_class_count / count - 1
                    train_fcs = np.append(train_fcs, np.repeat(train_fcs[start: start+count], repeat_num, axis=0), axis=0)
                    train_labels = np.append(train_labels, np.repeat(train_labels[start: start+count], repeat_num, axis=0), axis=0)
        train_dataset = DataSet(None, train_fcs, train_labels)

    if read_test:
        test_fcs = np.load(os.path.join(data_dir, config.test_feature_file))
        test_fcs = multiview(test_fcs, config.num_views)
        #test_fcs = maxpooling(test_fcs)
        test_labels = np.load(os.path.join(config.data, config.test_label_file))
        test_labels = onehot(test_labels)
        if config.enrich_data:
            test_fcs, test_labels = roll_enrich(test_fcs, test_labels, roll_number)
            print(test_fcs.shape, test_labels.shape)
        test_dataset = DataSet(None, test_fcs, test_labels)

    print("read data finished")
    return base.Datasets(train=train_dataset, test=test_dataset, validation=None)

def roll_enrich(fcs, labels, roll_number):
    new_fcs = np.concatenate([np.roll(fcs, i, axis=1)  for i in range(roll_number)])
    new_labels = np.concatenate([np.copy(labels) for _ in range(roll_number)])
    return new_fcs, new_labels

def labels_statistic(labels):
    label_range = {}
    for i in range(labels.shape[0]):
        if labels[i] not in label_range:
            label_range[labels[i]] = [i, 1] #start position, count
        else:
            label_range[labels[i]][1] += 1 #add count
    return label_range

def multiview(fcs, n_views=12):
    fcs2 = np.zeros(shape=[fcs.shape[0], n_views, fcs.shape[2]])
    for i in range(len(fcs)):
        #firstfc = np.reshape(fcs[i][0], [1, fcs.shape[2]])
        #fcs2[i] = np.repeat(firstfc, n_views, axis=0)
        fcs2[i] = fcs[i][:n_views]
        #fcs2[i] = np.roll(fcs2[i], random.randint(0,11), axis=0)
    return fcs2

def maxpooling(fcs):
    return np.max(fcs, axis=1)

def onehot(labels):
    label_count = np.shape(labels)[0]
    labels2 = np.zeros(shape=[label_count, config.num_classes]) # TODO shape=[batch_size, n_classes]
    labels2[np.arange(label_count), labels] = 1
    return labels2

def _fake_write_data(data_dir):
    train_fcs = np.zeros(shape=[2, 12, 13])
    train_labels = np.ones(shape=[2])

    test_fcs = np.ones(shape=[2,12,13])
    test_labels = np.zeros(shape=[2])

    np.save(os.path.join(data_dir, config.train_feature_file[ : config.train_feature_file.find('.')]), train_fcs)
    np.save(os.path.join(data_dir, config.train_label_file[ : config.train_label_file.find('.')]), train_labels)

    np.save(os.path.join(data_dir, config.test_feature_file[ : config.test_feature_file.find('.')]), test_fcs)
    np.save(os.path.join(data_dir, config.test_label_file[ : config.test_label_file.find('.')]), test_labels)

def run_readdata_demo(data_dir):
    model_data = read_data(data_dir)
    train_data = model_data.train
    test_data = model_data.test
    fc1, label1 = test_data.next_batch(2)
    print("fc1:", np.shape(fc1))
    print("label1:", label1)
    print("shape:", np.shape(label1))
    print("target", get_target_labels(label1))

def label2sequence(label_onehot):
    label = np.argmax(label_onehot) + 1
    sequence = []
    for i in range(1, np.shape(label_onehot)[0]+1):
        if label != i:
            sequence.append(2*i)
        else:
            sequence.append(2*i-1)
    return np.array(sequence)

def get_target_labels(seq_labels):
    target_labels = []
    for i in range(np.shape(seq_labels)[0]): #loop batch_size
        for j in range(np.shape(seq_labels)[1]): #loop label
            if seq_labels[i][j] % 2 == 1:
                target_labels.append((seq_labels[i][j]+1)/2)
                break
    return target_labels

def read_class_yes_embedding(data_dir):
    yes_embedding = np.load(config.class_yes_feature_file)
    class_embedding = np.zeros([81, yes_embedding.shape[1]]) # 81=2*classes+1
    class_embedding[1::2] = yes_embedding
    return class_embedding


if __name__ == '__main__':
    read_data("", read_test=False, n_views=2)
    #run_readdata_demo(config.data)
    # label_onehot = np.zeros([40])
    # label_onehot[2] = 1
    # print label2sequence(label_onehot)