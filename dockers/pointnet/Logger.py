from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pandas
import os

class Logger_Entry:
    def __init__(self, times = None, data = None, epochs = None, offset = 0):
        self.begin_time = time.time()
        self._times = times or []
        self._data = data or []
        self._epochs = epochs or []
        self.offset = 0
   
    def add(self, dato, epoch):
        self._data.append(dato)
        self._epochs.append(epoch)
        self._times.append(time.time() -self.offset)
    
    @property
    def times(self):
        return [t - self.begin_time for t in self._times]
    @property
    def data(self):
        return self._data
        
class Logger:
    def __init__(self, name):
        self.data = {}
        self.name = name
        
    def log(self, to_log, epoch, name):
        if name not in self.data:
            self.data[name] = Logger_Entry()
        self.data[name].add(to_log, epoch)
        
    def plot(self, names=None, plot_title="Plot", x_axis="time", y_axis="data", save=True, dest="."):
        if names == None:
            names = self.data.keys()
        names = list(names)
        plt.title(plot_title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        for name in names:
            entry = self.data[name]
            plt.plot(entry.times, entry.data)
        plt.legend(names)
        if save:
            plt.savefig(os.path.join(dest,self.name + ".png"))
        else:
            plt.show()
        plt.clf()
    
    def save(self, dest="."):
        for name in self.data.keys():
            entry = self.data[name]
            data = {'time': entry._times, name : entry.data, 'epochs':entry._epochs}
            dataframe = pandas.DataFrame(data=data)
            dataframe = dataframe[['time', name, 'epochs']]
            dataframe.to_csv(os.path.join(dest,"{}_{}.csv".format(self.name, name)))
            
    def load(self, file_list, epoch=None):
        self.data = {}
        for file in file_list:
            data = pandas.read_csv(file)
            
            if epoch:
                data = data[data['epochs']<=epoch]

            name = list(data)[2]
            self.data[name] = Logger_Entry(times=list(data['time']), data=list(data[name]), epochs=list(data['epochs']))
            self.data[name].begin_time = data['time'].min()
            self.data[name].offset = time.time() - data['time'].max()
                
    
    
        
        