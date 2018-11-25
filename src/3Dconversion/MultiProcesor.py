from __future__ import print_function
from multiprocessing import Process, Lock

class MultiProcesor:
    def __init__(self, files, n_threads, log_file, categories, split, max_size_of_file, proccess_function, write_function):
        self.files = files
        self.n_threads = n_threads
        self.log_file = log_file
        self.lock = Lock()
        self.max_size = max_size_of_file
        self.proccess_function = proccess_function
        self.write_function = write_function
        self.categories = categories
        self.split = split
        
    def run(self, args):
        size = len(self.files) // self.n_threads
        pool = []
        if len(self.files) > self.n_threads *2:
            for i in range(self.n_threads-1):
                p = Process(target=self.run_conversion, args=(self.files[i*size:(i+1)*size], i, self.lock, self.log_file, args))
                p.start()
            p = Process(target=self.run_conversion, args=(self.files[(args.t-1)*size:], self.n_threads-1, self.lock, self.log_file, args))
            p.start()
            pool.append(p)
            for p in pool:
                p.join()
        else:
            self.run_conversion(self.files, 0, self.lock, self.log_file, args)
        
    
    def run_conversion(self, files, id, lock, log_file, args):
        datasets = ["train","test", "val"]
        self.log("Starting thread {} on {} files.".format(id, len(files)), lock, log_file)
        buffer = [[] for _ in range(len(datasets))]
        buffer_cats = [[] for _ in range(len(datasets))]
        splitss = [0] * len(datasets)
        logging_frequency = 50
        buffer_written = [0] * len(datasets)
        for i in range(len(files)):
            filename = files[i]
            if i>0 and i%logging_frequency == 0:
                self.log("Thread {} is {}% done.".format(id,float(i)/len(files)*100), lock, log_file)
            splited = files[i].split('/')
            for directory in splited:
                if directory in self.split:
                    split_index = self.split[directory]
                    splitss[split_index]+=1
                    break
            for directory in splited:
                if directory in self.categories:
                    buffer_cats[split_index].append(self.categories[directory])
                    break
            
            buffer[split_index].append(self.proccess_function(filename,args))
            
            for j in range(len(datasets)):
                if len(buffer_cats[j])  == self.max_size:
                    self.log("Thread {} started saving {} {}-th file.".format(id,datasets[j], buffer_written[j]), lock, log_file)
                    self.write_function(buffer[j], buffer_cats[j], datasets[j], id, buffer_written[j], args)
                    del buffer[j]
                    buffer.insert(j, [])
                    del buffer_cats[j]
                    buffer_cats.insert(j, [])
                    self.log("Thread {} ended saving {} {}-th file.".format(id, datasets[j], buffer_written[j]), lock, log_file)
                    buffer_written[j] += 1
        for j in range(len(datasets)):
            if len(buffer_cats[j])  > 0:
                self.log("Thread {} started saving {} {}-th file.".format(id,datasets[j], buffer_written[j]), lock, log_file)
                self.write_function(buffer[j], buffer_cats[j], datasets[j], id, buffer_written[j], args)
                self.log("Thread {} ended saving {} {}-th file.".format(id, datasets[j], buffer_written[j]), lock, log_file)
    
    def log(self, message, lock, log):
        lock.acquire()
        with open(log, 'a') as f:
            print(message, file = f)
        lock.release()
        
    