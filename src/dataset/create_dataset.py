'''
Created on 25. 7. 2018

@author: Mirek
'''
from .parse_dataset import Model, Room, Data
import pickle
import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self, data, maxsize, shuffle_batches=True):
        
        room_cats = self._make_index_mapping(data.unique_room_types)
        model_cats = self._make_index_mapping(data.unique_model_types)
        
        self._sequences = np.zeros([len(data.rooms), maxsize, 7], np.float32)
        self._labels = np.zeros([len(data.rooms), 3], np.float32)

        for room in range(len(data.rooms)):
            random_index = np.random.randint(len(room.models))
            for model in range(maxsize):
                if model != random_index:
                    if len(data.rooms[room].models) > model:
                        self._sequences[room, model, 0] = model_cats[data.rooms[room].models[model].type]
                        self._sequences[room, model, 1:] = data.rooms[room].models[model].bbox['min'] + data.rooms[room].models[model].bbox['min']
                else:
                    self._labels[room,:] = data.rooms[room].models[model].bbox['min']

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._sequences)) if self._shuffle_batches else np.arange(len(self._sequences))

    def _make_index_mapping(self, sett):
        sett = list(sett)
        mapping = {}
        i = 1
        for category in sett:
            mapping[category] = i
            i=i+1
        return mapping


    @property
    def sequences(self):
        return self._sequences

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._sequences, self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._sequences[batch_perm], self._labels[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sequences)) if self._shuffle_batches else np.arange(len(self._sequences))
            return True
        return False


if __name__ == '__main__':
    import argparse

    np.random.seed(42)
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".", type=str, help="Path to pickled data")
    parser.add_argument("--val",default=0.2, type=float, help="size of te validation set")
    args = parser.parse_args()
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    dataset = Dataset(data, 10, args.val)
    