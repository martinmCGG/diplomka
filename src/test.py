from multiprocessing import Process
import math

def compute_nonsense(i):
    print("STARTING {}".format(i))
    for j in range(500000):
        a = math.sin(j)
        b = math.sin(j)
        c = a * b
    print("ENDING {}".format(i))
    
    
def run_threads(n):
    pool = []
    for i in range(n):
        p = Process(target=compute_nonsense, args=(i,))
        pool.append(p)
        p.start()
    p = Process(target=compute_nonsense, args=(n+1,))
    p.start()
    pool.append(p)
    for p in pool:
        p.join()

def cleanup():
    print("CLEANUP")


if __name__ == '__main__':
    run_threads(5)
    cleanup()