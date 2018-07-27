'''
Created on 25. 7. 2018

@author: Mirek
'''
from parse_dataset import Model, Data, Room
import pickle
import os
import numpy as np

class Dataset:
    def __init__(self, data, maxsize, shuffle_batches=True):
        
        self.data = data
        
        room_cats = self._make_index_mapping(data.unique_room_types)
        model_cats = self._make_index_mapping(data.unique_model_types)
        
        self._categories = np.zeros([len(data.rooms), maxsize],np.int32)
        self._sequences = np.zeros([len(data.rooms), maxsize, 6], np.float32)
        self._labels_categories = np.zeros([len(data.rooms)], np.int32)
        self._labels = np.zeros([len(data.rooms), 3], np.float32)
        
        self._sequence_lengths = np.zeros([len(data.rooms)], np.int32)


        offset = 0
        for room in range(len(data.rooms)):
            size_of_room = len(data.rooms[room].models)
            random_index = np.random.randint(len(data.rooms[room].models))
            for model in range(maxsize):
                if size_of_room > model:
                    modell = data.rooms[room].models[model]
                    if model != random_index:
                        if modell != None:
                            self.categories[room, model+offset] = model_cats[modell.type]
                            self._sequences[room, model+offset,:] = modell.bbox['min'] + modell.bbox['max']
                    else:
                        if modell !=None:
                            offset = -1
                            self._labels_categories[room] = model_cats[modell.type]
                            self._labels[room,:] = modell.bbox['min']
                else:
                    break
            self._sequence_lengths[room] = size_of_room - 1 
            
        #print(self._sequences)
        #print(self._labels)
        
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
    def categories(self):
        return self._categories
    @property
    def labels(self):
        return self._labels
    @property
    def labels_categories(self):
        return self._labels_categories
   
    @property
    def sequence_lengths(self):
        return self._sequence_lengths

    def all_data(self):
        return self._sequences, self._labels, self._categories, self._labels_categories, self._sequence_lengths

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._sequences[batch_perm], self._labels[batch_perm], self._categories[batch_perm],self._labels_categories[batch_perm], self._sequence_lengths[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sequences)) if self._shuffle_batches else np.arange(len(self._sequences))
            return True
        return False

    def get_number_of_categories(self):
        return len(self.data.unique_model_types)
    
if __name__ == '__main__':
    import argparse
    np.random.seed(42)
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to pickled data")
    args = parser.parse_args()
    
    with open(os.path.join(args.folder,"train.pickle"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.folder,"val.pickle"), 'rb') as f:
        val_data = pickle.load(f)
    train = Dataset(train_data, 10)
    val = Dataset(val_data, 10)
    
    