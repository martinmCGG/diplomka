import numpy as np
import scipy.misc
import os


def save_batch_as_image(images, name):
    proccesed = []
    for i in range(len(images[0])):
        proccesed.append(_make_one_image([x[i] for x in images]))
    concatenated = np.concatenate(proccesed,axis=1)
    scipy.misc.imsave(os.path.join('images', name+'.jpg'), concatenated)
    
    
def _make_one_image(images):
    concatenated = np.concatenate(images,axis=0)    
    image = np.where(concatenated>0,255,0)
    return image