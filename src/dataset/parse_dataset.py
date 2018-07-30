'''
Created on 25. 7. 2018

@author: Mirek
'''
import json
import os
import pickle


class Room:
    def __init__(self):   
        self.types = []
        self.models = []
        self.indexes = []
        self.id = ""
        self.house_id = ""
        self.bbox = {}
        
class Model:

    def __init__(self):
        self.bbox = {"min":0,"max":0}
        self.type = ""
        self.id = "" 
    def __repr__(self):
        return self.id + " " + self.type
        
        
class Data:

    zaporne = 0
    kladne = 0
    ignored_categories = ["otherprop"]

    def load_types(self):
        model_types = {}
        with open(os.path.join(self.root_folder,"ModelCategoryMapping.csv"),'r') as f:
            for line in f:
                splited = line.split(',')
                model_types[splited[1]] = splited[5]
        return model_types
    
    
    def __init__(self, root_folder):
        self.unique_room_types = []
        self.unique_model_types = []
        self.root_folder = root_folder

        self.rooms = []
        if root_folder != None:
            self.model_types = self.load_types()
            houses = os.listdir(os.path.join(root_folder,"house")) 
            
            i=0
            for house_id in houses:
                print(house_id)
                self.rooms = self.rooms + self._load_house(house_id)
                #if i>10:
                #    break
                #i+=1
        print(-1*self.zaporne,  self.kladne)
    
    def _load_house(self, house_id):
        with open(os.path.join(self.root_folder,'house',house_id,"house.json"),'r') as f:
            json_house = f.readline()
        parsed = json.loads(json_house)
        
        rooms = []
        models = []
            
        for level in parsed['levels']:
            for entry in level['nodes']:
                if entry['type'] == 'Object':
                    model = Model()
                    model.id = entry['modelId']
                    model.bbox = entry['bbox']
                    model.type = self.model_types[model.id]
                    if model.type not in self.ignored_categories and model.type not in self.unique_model_types:
                        self.unique_model_types.append(self.model_types[model.id])                    
                    models.append(model)
                
                elif entry['type'] == 'Room':
                    if 'nodeIndices' in entry: #skip empty rooms
                        room = Room()
                        room.id = entry['id']
                        room.house_id = house_id
                        room.indexes = entry['nodeIndices']
                        room.types = entry['roomTypes']
                        room.bbox = entry["bbox"]
                        for t in entry['roomTypes']:
                            if t not in self.unique_room_types:
                                self.unique_room_types.append(t)
                        rooms.append(room)
                    models.append(None)
                else:
                    models.append(None)
                    

        for room in rooms:
            for node in room.indexes:
                model = models[node]
                if model!= None and model.type not in self.ignored_categories:
                    #Filter the models which are not inside the room
                    if self._is_inside(room.bbox, model.bbox):
                        self.kladne+=1
                        room.models.append(models[node])
                        model.bbox = self._subtract_bbox(model.bbox, room.bbox["min"])
                    else:
                        self.zaporne +=1
            room.bbox = self._subtract_bbox(room.bbox, room.bbox["min"])
        
        #Filter out empty rooms
        return [x for x in rooms if x.models]

    def _is_inside(self, big, small):
        istrue = True
        for i in [0,2]:
            istrue = istrue and big["min"][i] <= small["min"][i]
        for i in [0,2]:
            istrue = istrue and big["max"][i] >= small["max"][i]
        return istrue
    
    
    def write_to_file(self, filename, val):
        valset = Data(None)
        valset.unique_model_types = self.unique_model_types
        valset.unique_room_types = self.unique_room_types
        border = int(len(self.rooms) * val)
        
        valset.rooms = self.rooms[:border]
        self.rooms = self.rooms[border:]
        
        with open(os.path.join(filename,"val.pickle"), 'wb') as f:
            pickle.dump(valset,f)
        with open(os.path.join(filename,"train.pickle"), 'wb') as f:
            pickle.dump(self,f)
            
    def _subtract_bbox(self, bbox, sub):
        rt = {}
        for m in ["min","max"]:
            rt[m] = [bbox[m][i] - sub[i] for i in range(3)]
        return rt
           
        
if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to data")
    parser.add_argument("--val", default=0.2, type=float, help="Size of validation set")
    
    args = parser.parse_args()
    
    dataset = Data(args.folder)
    dataset.write_to_file(args.folder, args.val)
    