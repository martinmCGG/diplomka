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
        
    def plot(self, names=None, plot_title="Plot", x_axis="time", y_axis="data", save=True, dest=".", maxtime=None, mintime=None, epoch=False):
        if names == None:
            names = self.data.keys()
        names = sorted(list(names))

        for name in names:
            entry = self.data[name]
            
            df = {'time': entry.times, name : entry.data, 'epoch' : entry._epochs}
            df = pandas.DataFrame(data=df)
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

def load_and_plot(directory, regex, name="Plot", maxtime = None, mintime=None, epoch=False):
    logger = Logger(name)
    import re 
    all_files = []
    all_names = []
    models = os.listdir(directory)
    for model in models:
        model_directory = os.path.join(directory, model, "out")
        if os.path.exists(model_directory):
            csv_files = os.listdir(model_directory)
            csv_files = [x for x in csv_files if re.match(regex, x)]
            all_names += csv_files
            csv_files = [os.path.join(model_directory,x) for x in csv_files]
            all_files += csv_files
    logger.load(all_files, names = all_names)
    logger.plot(dest=directory, maxtime = maxtime, mintime=mintime, epoch=epoch)

if __name__ == '__main__':
    load_and_plot('.',".*_acc_eval_accuracy.csv", maxtime=None, mintime=None, epoch=True)
        
        

        
        
        
        