from __future__ import print_function
from collections import namedtuple
import ConfigParser


def Parse(value):
    try:
        return int(value)
    except ValueError:
        if value in ['False','false']:
            return False
        if value in ['True','true']:
            return True
        return value
    


class config:
    
    def __init__(self, ini_file = 'config.ini'):
        self.cp = ConfigParser.RawConfigParser()
        self.cp.read(ini_file) 
        self.config_to_dict()
    
    def config_to_dict(self):
        self.dictionary = {}
        for section in self.cp.sections():
            for key, value in self.cp.items(section):
                self.dictionary[key] = Parse(value)
        self.dictionary = namedtuple("config", self.dictionary.keys())(*self.dictionary.values())
        print(self.dictionary)

def get_config():
    return config().dictionary