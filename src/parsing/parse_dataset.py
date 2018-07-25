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
        
class Model:

    def __init__(self):
        self.bbox = {min:0,max:0}
        self.type = ""
        self.id = "" 
    def __repr__(self):
        return self.id + " " + self.type
        
        
class Dataset:
    
    def load_types(self):
        model_types = {}
        with open(os.path.join(self.root_folder,"ModelCategoryMapping.csv"),'r') as f:
            for line in f:
                splited = line.split(',')
                model_types[splited[1]] = splited[5]
        return model_types
    
    
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.model_types = self.load_types()
        self.rooms = []
        houses = os.listdir(os.path.join(root_folder,"house")) 
        for house_id in houses:
            print(house_id)
            self.rooms = self.rooms + self._load_house(house_id)
            #break
    
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
                    models.append(model)
                
                elif entry['type'] == 'Room':
                    if 'nodeIndices' in entry: #skip empty rooms
                        room = Room()
                        room.id = entry['modelId']
                        room.indexes = entry['nodeIndices']
                        room.types = entry['roomTypes']
                        rooms.append(room)
                    models.append(None)
                else:
                    models.append(None)
        for room in rooms:
            for node in room.indexes:
                room.models.append(models[node])


        return rooms

    def write_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.rooms,f)
        
        
if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to data")
    parser.add_argument("--output", default="data.txt", type=str, help="Path to output file")
    
    args = parser.parse_args()
    
    dataset = Dataset(args.folder)
    dataset.write_to_file(args.output)
    