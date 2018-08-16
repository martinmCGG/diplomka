'''
Created on 25. 7. 2018

@author: Mirek
'''
from parse_dataset import Model, Data, Room
import pickle
import os
import numpy as np
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)

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
        batch = [x[batch_perm] for x in self.data_created]
        return batch
    
    def epoch_finished(self, batch_size):
        if (len(self.permutation)//batch_size) % 100 == 0: 
            print('{} batches left.'.format(len(self.permutation)//batch_size))
        if len(self.permutation) < batch_size:
            self.permutation = np.random.permutation(len(self.data_created[0])) if self.shuffle_batches else np.arange(len(self.data_created[0]))
            
            return True
        
        return False
    
    def is_last_batch(self, batch_size):
        return len(self.permutation) < batch_size*2
    
    def get_number_of_categories(self):
        return len(self.data.unique_model_types)
    
    def _make_index_mapping(self, sett):
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

    def __init__(self, data, image_size, type_of_prediction, shuffle_batches=True):
        self.image_size = image_size 
        self.size = data.count_models()
        print("Creating {} examples...".format(self.size))
        self.images = np.zeros([self.size, image_size, image_size], np.int32)
        
        if type_of_prediction == 'coordinates':
            self.labels = np.zeros([self.size,2], np.int32)
            self.label_cats = np.zeros([self.size], np.int32)
        elif type_of_prediction == 'map':
            self.labels = np.zeros([self.size, image_size, image_size], np.int32)
            self.label_cats = np.zeros([self.size, image_size, image_size], np.int32)
        
        data_created = [self.images, self.labels, self.label_cats]
        Dataset.__init__(self, data, data_created, shuffle_batches)
        
        index = 0
        for room in data.rooms:
            for model in room.models:
                proom = self._proccess_room_model(room,model,type_of_prediction)
                self.images[index,:,:] = proom[0]
                if type_of_prediction == 'coordinates':
                    self.labels[index,0] = proom[1][0]
                    self.labels[index,1] = proom[1][1]
                    self.label_cats[index] = proom[2]
                elif type_of_prediction == 'map':
                    self.labels[index,:,:] = proom[1]
                    self.label_cats[index,:,:] = proom[2]
                index+=1
        
    def _proccess_room_model(self, room, missing_model, type_of_prediction):
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
            if model != missing_model:
                for i in range(minx, maxx):
                    for j in range(minz, maxz):
                        image[i,j] = category
            else:
                if type_of_prediction == 'coordinates':
                    label = [minx + (maxx - minx)/2, minz + (maxz - minz)/2]
                    model_category = category
                
                elif type_of_prediction == 'map':
                    label = np.zeros((self.image_size,self.image_size), np.int32)
                    for i in range(minx, maxx):
                        for j in range(minz, maxz):
                            label[i,j] = 1
                    model_category = np.zeros((self.image_size,self.image_size), np.int32)
                    for i in range(0, maxx-minx):
                        for j in range(0, maxz-minz):
                            model_category[i,j] = category   
        return image, label, model_category
    
        

class RoomClassDataset(Dataset):
    def __init__(self, data, image_size, shuffle_batches=True):
        self.image_size = image_size
        
        data.rooms = [x for x in data.rooms if len(x.types)==1]
        
        self.images = np.zeros([len(data.rooms), image_size, image_size], np.int32)
        self.labels = np.zeros([len(data.rooms)], np.int32)
        
        data_created = [self.images, self.labels]
        Dataset.__init__(self, data, data_created, shuffle_batches)
        
        for r in range(len(data.rooms)):
            room = data.rooms[r]
            proom = self._proccess_room(room)
            self.images[r,:,:] = proom
            self.labels[r] = self.room_cats[room.types[0]]
        self.data_created = [self.images, self.labels]
        #print(self.data_created)
        
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
        return image
        
    