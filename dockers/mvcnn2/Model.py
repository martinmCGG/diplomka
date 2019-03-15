import torch
import torch.nn as nn
import os
import glob


class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name


    def save(self, file):
        torch.save(self.state_dict(), file)

    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        
    def load(self, file):    
        if not os.path.exists(file):
            raise IOError("{} directory does not exist in {}".format(self.name, path))
        self.load_state_dict(torch.load(file))


