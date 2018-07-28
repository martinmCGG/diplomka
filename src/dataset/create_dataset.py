'''
Created on 25. 7. 2018

@author: Mirek
'''
from parse_dataset import Model, Data, Room
import pickle
import os
import numpy as np


class Dataset:
    def __init__(self, data, data_created, shuffle_batches=True):
        self.data = data
        self.room_cats = self._make_index_mapping(data.unique_room_types)
        self.model_cats = self._make_index_mapping(data.unique_model_types)
        
        self.shuffle_batches = shuffle_batches
        self.permutation = np.random.permutation(len(self.sequences)) if self.shuffle_batches else np.arange(len(self.sequences))
    
        self.data_created = data_created
        
    def all_data(self):
        return self.data_created

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self.permutation))
        batch_perm, self.permutation = self.permutation[:batch_size], self.permutation[batch_size:]
        return [x[batch_perm] for x in self.data_created]
    
    def epoch_finished(self):
        if len(self.permutation) == 0:
            self.permutation = np.random.permutation(len(self.sequences)) if self.shuffle_batches else np.arange(len(self.sequences))
            return True
        return False
    
    def get_number_of_categories(self):
        return len(self.data.unique_model_types)
    
    def _make_index_mapping(self, sett):
        sett = list(sett)
        mapping = {}
        i = 1
        for category in sett:
            mapping[category] = i
            i=i+1
        return mapping


class RnnDataset(Dataset):
    
    
    
    def __init__(self, data, maxsize, shuffle_batches=True):
        self.maxsize = maxsize
        
        self.categories = np.zeros([len(data.rooms), maxsize],np.int32)
        self.sequences = np.zeros([len(data.rooms), maxsize, 6], np.float32)
        self.labels_categories = np.zeros([len(data.rooms)], np.int32)
        self.labels = np.zeros([len(data.rooms), 3], np.float32)
        self.sequence_lengths = np.zeros([len(data.rooms)], np.int32)
        
        
        data_created = [self.sequences,self.labels, self.categories, self.labels_categories,  self.sequence_lengths]
        
        Dataset.__init__(self, data, data_created, shuffle_batches)
        self._create_data()
        
    

    
class ConvDataset(Dataset):
    def __init__(self, data, size_of_pixel, shuffle_batches=True):
        self.size_of_pixel = size_of_pixel
        
        
        
    
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
    
    