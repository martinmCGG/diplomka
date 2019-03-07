from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pandas
import os
import glob
import numpy as np
from Evaluation_tools import collect_files


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
     
    def get_as_pandas(self, name):
        entry = self.data[name]  
        df = {'time': entry.times, name : entry.data, 'epoch' : entry._epochs}
        df = pandas.DataFrame(data=df)
        return df
        
    def get_epoch_times(self,names=None, dest='.', csv=False):
        if names == None:
            names = self.data.keys()
        names = sorted(list(names))
        extension = 'csv' if csv else 'txt'
        with open(os.path.join(dest,'training_times.{}'.format(extension)), 'w') as f:
            if csv:
                print('{};{};{};{}'.format('name','epochs','time','time_per_epoch'), file=f)
            for name in names:
                df = self.get_as_pandas(name)
                max_epoch = df['epoch'].max()
                time = df['time'].max()/60
                if csv:
                    to_print = '{};{};{:.2f};{:.2f}'.format(name, max_epoch, time, time/max_epoch)
                else:
                    to_print = '{}{}: Trained for {} epochs for {:.2f} minutes. One epoch took {:.2f} minutes'
                    to_print = to_print.format(name,(14-len(name))*" ", max_epoch, time, time/max_epoch)
                print(to_print, file=f)
                print(to_print)
        
    def plot(self, names=None, plot_title="Plot", x_axis="time", y_axis="data", save=True, dest=".", maxtime=None, mintime=None, epoch=False):
        if names == None:
            names = self.data.keys()
        names = sorted(list(names))
        for name in names:
            df = self.get_as_pandas(name)
            if maxtime:
                df = df[df['time']<=maxtime]
            if mintime:
                df = df[df['time']>=mintime]
            #times = list(df['time']/60)
            times = list(df['time'])
            if epoch:
                df = df.groupby(['epoch']).mean()  
                times = range(df.shape[0])
                x_axis = 'epochs'
            data = list(df[name])
            plt.plot(times, data)
        plt.title(plot_title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
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
            
    def load(self, file_list, epoch=None, names=None):
        self.data = {}
        for i in range(len(file_list)):
            file = file_list[i]
            data = pandas.read_csv(file)
            if epoch:
                data = data[data['epochs']<=epoch]
            if names:
                name = names[i]
            else:
                name = list(data)[2]
            data_name = list(data)[2]
            
            self.data[name] = Logger_Entry(times=list(data['time']), data=list(data[data_name]), epochs=list(data['epochs']))
            self.data[name].begin_time = data['time'].min()
            self.data[name].offset = time.time() - data['time'].max()
    

def find_name(dir):
    files = collect_files(dir)
    if len(files) > 0:
        return open(files[0],'r').readlines()[0].strip()
    else:
        return os.path.basename(dir)

def find_and_load(directory, regex, name='Plot'):
    import re
    all_files = []
    all_names = []  
    models = glob.glob(os.path.join(directory, '**','*.csv'),recursive=True)
    logger = Logger(name)
    for model in models:
        if re.match(regex, model):
            print(model)
            all_files.append(model)
            all_names.append(find_name(os.path.dirname(model)))
    logger.load(all_files, names = all_names)
    return logger


if __name__ == '__main__':
    #logger = find_and_load('.',".*rotnet.*.*_acc_eval_accuracy.csv")
    #logger.plot(dest='.', maxtime=None, mintime=None, epoch=True)
    
    logger = find_and_load('.',".*_acc_train_accuracy.csv")
    logger.get_epoch_times(csv=True)
    logger.get_epoch_times(csv=False)  
        
        

        
        