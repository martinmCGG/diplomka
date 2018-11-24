from __future__ import print_function
from multiprocessing import Process, Lock

class MultiProccessor:
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
        if len(self.files) > self.n_threads *2:
            for i in range(self.n_threads-1):
                Process(target=self.run_conversion, args=(self.files[i*size:(i+1)*size], i, self.lock, self.log_file, args)).start()
            Process(target=self.run_conversion, args=(self.files[(args.t-1)*size:], self.n_threads-1, self.lock, self.log_file, args)).start()
        else:
            self.run_conversion(self.files, 0, self.lock, self.log_file, args)
    
    def run_conversion(self, files, id, lock, log_file, args):
        self.log("Starting thread {} on {} files.".format(id, len(files)), lock, log_file)
        buffer = []
        buffer_cats = []
        buffer_split = []
        logging_frequency = 50
        buffer_written = 0
        for i in range(len(files)):
            filename = files[i]
            if i>0 and i%logging_frequency == 0:
                self.log("Thread {} is {}% done.".format(id,float(i)/len(self.files)*100), lock, log_file)
            splited = files[i].split('/')
            for directory in splited:
                if directory in self.categories:
                    buffer_cats.append(self.categories[directory])
                    break
            for directory in splited:
                if directory in self.split:
                    buffer_split.append(self.split[directory])
                    break
                
            buffer += (self.proccess_function(filename,args))
            if len(buffer_cats)  == self.max_size:
                self.log("Thread {} started saving {}-th file.".format(id, buffer_written), lock, log_file)
                self.write_function(buffer, buffer_cats, buffer_split, id, buffer_written, args)
                buffer = []
                buffer_cats = []
                buffer_split = []
                buffer_written += 1
                self.log("Thread {} ended saving {}-th file.".format(id, buffer_written), lock, log_file)
        if len(buffer_cats) > 0:
            self.write_function(buffer, buffer_cats,buffer_split, id, buffer_written, args)
    
    def log(self, message, lock, log):
        lock.acquire()
        with open(log, 'a') as f:
            print(message, file = f)
        lock.release()
        
    