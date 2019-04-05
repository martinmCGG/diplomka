from __future__ import print_function
import os

import numpy as np
import h5py as h5


def prepare(config):
    path2data = config.data
    path2save = config.output
    train_nFaces = np.zeros((1,), dtype=np.int32)
    train_faces = np.empty((0, 3), dtype=np.int32)
    train_vertices = np.empty((0, 3), dtype=np.float32)
    train_labels = np.empty((0,), dtype=np.int8)

    test_nFaces = np.zeros((1,), dtype=np.int32)
    test_faces = np.empty((0, 3), dtype=np.int32)
    test_vertices = np.empty((0, 3), dtype=np.float32)
    test_labels = np.empty((0,), dtype=np.int8)

    classes = sorted(os.listdir(path2data))
    class2label = {}
    
    label = 0
    for i, cl in enumerate(classes):
        
        if not os.path.exists(path2data + '/' + cl + '/train/'):
            continue
        class2label[cl] = np.int8(label)
        label+=1
        cl_train_filenames = sorted(os.listdir(path2data + '/' + cl + '/train/'))
        cl_test_filenames = sorted(os.listdir(path2data + '/' + cl + '/test/'))
        
        cl_train_filenames = [x for x in cl_train_filenames if x.split('.')[-1] == 'off']
        cl_test_filenames= [x for x in cl_test_filenames if x.split('.')[-1] == 'off']
        
        for j, shapefile in enumerate(cl_train_filenames):
            print(shapefile)
            with open(path2data + '/' + cl + '/train/' + shapefile, 'rb') as fobj:
                for k, line in enumerate(fobj):
                    try:
                        if k == 0 and line.strip() != 'OFF':
                            numVertices, numFaces, numEdges = map(np.int32, line[3:].split())
                            break
                        numVertices, numFaces, numEdges = map(np.int32, line.split())
                        break
                    except:
                        pass

                vrtcs = np.empty((numVertices, 3), dtype=np.float32)
                for k, line in enumerate(fobj):
                    vrtcs[k] = list(map(np.float32, line.split()))
                    if k == numVertices - 1:
                        break

                fcs = np.empty((numFaces, 3), dtype=np.int32)
                for k, line in enumerate(fobj):
                    fcs[k] = list(map(np.int32, line.split()))[1:]
                    if k == numFaces - 1:
                        break

            train_nFaces = np.hstack((train_nFaces, numFaces + train_nFaces[-1]))
            train_faces = np.vstack((train_faces, fcs + len(train_vertices)))
            train_vertices = np.vstack((train_vertices, vrtcs))
        train_labels = np.hstack((train_labels, class2label[cl]*np.ones(len(cl_train_filenames), dtype=np.int8)))

        for j, shapefile in enumerate(cl_test_filenames):
            with open(path2data + '/' + cl + '/test/' + shapefile, 'rb') as fobj:
                print(shapefile)
                for k, line in enumerate(fobj):
                    try:
                        if k == 0 and line.strip() != 'OFF':
                            numVertices, numFaces, numEdges = map(np.int32, line[3:].split())
                            break
                        numVertices, numFaces, numEdges = map(np.int32, line.split())
                        break
                    except:
                        pass

                vrtcs = np.empty((numVertices, 3), dtype=np.float32)
                for k, line in enumerate(fobj):
                    vrtcs[k] = list(map(np.float32, line.split()))
                    if k == numVertices - 1:
                        break

                fcs = np.empty((numFaces, 3), dtype=np.int32)
                for k, line in enumerate(fobj):
                    fcs[k] = list(map(np.int32, line.split()))[1:]
                    if k == numFaces - 1:   
                        break

            test_nFaces = np.hstack((test_nFaces, numFaces + test_nFaces[-1]))
            test_faces = np.vstack((test_faces, fcs + len(test_vertices)))
            test_vertices = np.vstack((test_vertices, vrtcs))
        test_labels = np.hstack((test_labels, class2label[cl]*np.ones(len(cl_test_filenames), dtype=np.int8)))
        
        print('{} - processed'.format(cl))

    with h5.File(path2save + '/data.h5', 'w') as hf:
        hf.create_dataset('train_nFaces', data=train_nFaces)
        hf.create_dataset('train_faces', data=train_faces)
        hf.create_dataset('train_vertices', data=train_vertices)
        hf.create_dataset('train_labels', data=train_labels)
        hf.create_dataset('test_nFaces', data=test_nFaces)
        hf.create_dataset('test_faces', data=test_faces)
        hf.create_dataset('test_vertices', data=test_vertices)
        hf.create_dataset('test_labels', data=test_labels)

    print('\nData is processed and saved to ' + path2save + '/data.h5')
    
    
