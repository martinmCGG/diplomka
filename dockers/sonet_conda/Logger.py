import matplotlib.pyplot as plt
import pandas   
import time
import math

class Logger_Entry:
    def __init__(self):
        self.begin_time = time.time()
        self.times = []
        self.data = []
    def add(self, dato):
        self.data.append(dato)
        self.times.append(time.time() - self.begin_time)
    
class Logger:
    def __init__(self):
        self.data = {}
        
    def log(self, to_log, name):
        if where_to_log not in self.data:
            self.data[name] = Logger_Entry()
        self.data[where_to_log].add(to_log)
        
    def plot(self,name):
        plt.plot(self.times, self.data)
        plt.show()
        
        
if __name__=='__main__':
    logger =Logger()
    for i in range(100):
        x = sin(i) * cos(i) + i*100
        logger.log(x,"sincos")
    logger.plot("sincos")