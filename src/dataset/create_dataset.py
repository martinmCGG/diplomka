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
        
        self.data_created = data_created
        
        self.shuffle_batches = shuffle_batches
        self.permutation = np.random.permutation(len(self.data_created[0])) if self.shuffle_batches else np.arange(len(self.data_created[0]))
    
        
        
    def all_data(self):
        return self.data_created

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self.permutation))
        batch_perm, self.permutation = self.permutation[:batch_size], self.permutation[batch_size:]
        return [x[batch_perm] for x in self.data_created]
    
    def epoch_finished(self):
        if len(self.permutation) == 0:
            self.permutation = np.random.permutation(len(self.data_created[0])) if self.shuffle_batches else np.arange(len(self.data_created[0]))
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
    
    def _create_data(self):
        offset = 0
        for room in range(len(self.data.rooms)):
            size_of_room = len(self.data.rooms[room].models)
            random_index = np.random.randint(len(self.data.rooms[room].models))
            for model in range(self.maxsize):
                if size_of_room > model:
                    modell = self.data.rooms[room].models[model]
                    if model != random_index:
                        if modell != None:
                            self.categories[room, model+offset] = self.model_cats[modell.type]
                            self.sequences[room, model+offset,:] = modell.bbox['min'] + modell.bbox['max']
                    else:
                        if modell !=None:
                            offset = -1
                            self.labels_categories[room] = self.model_cats[modell.type]
                            self.labels[room,:] = modell.bbox['min']
                else:
                    break
            self.sequence_lengths[room] = size_of_room - 1 
    
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

    def __init__(self, data, image_size, shuffle_batches=True):
        self.image_size = image_size
        
        self.images = np.zeros([len(data.rooms), image_size, image_size], np.int32)
        data_created = [self.images]
        Dataset.__init__(self, data, data_created, shuffle_batches)
        
        for r in range(len(data.rooms)):
            room = data.rooms[r]
            self.images[r,:,:] = self._proccess_room(room)
        print(np.shape(self.images))
            
        
        
        
    def _proccess_room(self, room):
        image = np.zeros((self.image_size,self.image_size), np.int32)
        bbmax = room.bbox["max"]
        x = bbmax[0]
        z = bbmax[2]
        maximum = max(x,z)
        pixel_size = self.image_size/maximum
        
        x = x * pixel_size
        z = z * pixel_size
        
        for model in room.models:
            category = self.model_cats[model.type]
            minx = int(model.bbox["min"][0] * pixel_size)
            maxx = int(model.bbox["max"][0] * pixel_size)
            minz = int(model.bbox["min"][2] * pixel_size)
            maxz = int(model.bbox["max"][2] * pixel_size)
            for i in range(minx, maxx):
                for j in range(minz, maxz):
                    image[i,j] = category
        
        np.savetxt('/home/krabec/'+room.id+'.txt', image)
        return image
    
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
    train = ConvDataset(train_data, 32)
    val = ConvDataset(val_data, 32)
    