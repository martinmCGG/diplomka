from __future__ import print_function
from multiprocessing import Process, Lock
import sys, os


class MultiProcesor:
    def __init__(self, files, n_threads, log_file, categories, split, dataset, proccess_function, write_function):
        self.files = files
        self.n_threads = min(n_threads, len(files)//20 + 1)
        self.log_file = log_file
        self.lock = Lock()
        self.proccess_function = proccess_function
        self.write_function = write_function
        self.categories = categories
        self.split = split
        self.dataset = dataset
        
    def run(self, args):
        size = len(self.files) // self.n_threads
        pool = []
        for i in range(self.n_threads-1):
            p = Process(target=self.run_conversion, args=(self.files[i*size:(i+1)*size], i, args))
            p.start()
            pool.append(p)
        if self.files[(self.n_threads-1)*size:]:
            p = Process(target=self.run_conversion, args=(self.files[(self.n_threads-1)*size:], self.n_threads-1, args))
            p.start()
            pool.append(p)
        for p in pool:
            p.join()
      
    def run_conversion(self, files, id, args):
        datasets = ["train","test", "val"]
        self.log("Starting thread {} on {} files.".format(id, len(files)))
        buffer = [[] for _ in range(len(datasets))]
        buffer_cats = [[] for _ in range(len(datasets))]
        splitss = [0] * len(datasets)
        logging_frequency = 100
        for i in range(len(files)):
            filename = files[i]
            if i>0 and i%logging_frequency == 0:
                self.log("Thread {} is {}% done.".format(id,float(i)/len(files)*100))
            try:
                file_id = get_file_id(filename, self.dataset)
                split_index = self.split[file_id]
                buffer[split_index].append(self.proccess_function(filename, self.dataset, args))
                splitss[split_index]+=1
                buffer_cats[split_index].append(self.categories[file_id])   
            except:
                e = sys.exc_info()
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                self.log("Exception occured in thread {}. Failed to proccess file {}".format(id, filename))
                self.log("Exception: Type: {} File: {} Line: {}".format(exc_type, fname, exc_tb.tb_lineno))                  
                    
        for j in range(len(datasets)):
            if len(buffer_cats[j])  > 0:
                self.log("Thread {} started saving {}.".format(id,datasets[j]))
                self.write_function(buffer[j], buffer_cats[j], datasets[j], id, args)
                self.log("Thread {} ended saving {}.".format(id, datasets[j]))
        self.log("Ending thread {}.".format(id))
        
    def log(self, message):
        self.lock.acquire()
        with open(self.log_file, 'a') as f:
            print(message, file = f)
        self.lock.release()

def get_file_id(file, dataset):
    if dataset == "shapenet":
        return file.split('/')[-3]
    elif dataset == "modelnet":
        return file.split('/')[-1].split('.')[-2]
    