from __future__ import print_function
from collections import namedtuple
try:
    import ConfigParser as cp
except:
    import configparser as cp

def Parse(value):
    if value in ['False','false']:
        return False
    if value in ['True','true']:
        return True
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass  
    return value 

class config:
    
    def __init__(self, ini_file, data_size=0):
        self.cp = cp.RawConfigParser()
        self.cp.read(ini_file) 
        self.config_to_dict(data_size=data_size)
    
    def config_to_dict(self, data_size=0):
        self.dictionary = {}
        for section in self.cp.sections():
            for key, value in self.cp.items(section):
                self.dictionary[key] = Parse(value)
                if section == 'ITER_PARAMETERS' and data_size:
                    value = epoch_to_iters(Parse(value), self.dictionary['batch_size'], data_size )
                    print(key, value)
                    self.dictionary[key] = value
        print(self.dictionary)
    
    def get_named_tuple(self):
        return namedtuple("config", self.dictionary.keys())(*self.dictionary.values())
    
    def prepare_caffe_files(self,file):
        newfile = []
        with open(file, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip()
            if ':' in line:
                key = line.split(':')[0]
                if key in self.dictionary:
                    line = '{}: {}'.format(key, self.dictionary[key])
            newfile.append(line)
        with open(file, 'w') as f:
            for line in newfile:
                print(line, file=f)
        
            
def get_config(ini_file = 'config.ini'):
    return config(ini_file).get_named_tuple()

def prepare_solver_file(ini_file = 'config.ini', data_size=0):
    cfg = config(ini_file, data_size=data_size)
    solver = cfg.get_named_tuple().solver
    cfg.prepare_caffe_files(solver)
def add_to_config(config, key, value):
    new_dict = config._asdict()
    new_dict[key] = value
    return namedtuple("config", new_dict.keys())(*new_dict.values())

def epoch_to_iters(epochs, batch_size, data_size):
    return epochs * data_size / batch_size

    
    