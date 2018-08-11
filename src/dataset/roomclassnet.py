import numpy as np
import tensorflow as tf
from create_dataset import Dataset, RoomClassDataset
from parse_dataset import Data, Room, Model
import pickle
import math

class Network:
    def __init__(self, threads, seed=0):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))
        
        
    def construct(self, args):
        with self.session.graph.as_default():        
            
            self.images = tf.placeholder(tf.int32, [None, args.room_size,args.room_size], name="images")
            self.labels = tf.placeholder(tf.float32, [None, args.labels_size], name="labels")
            
            #embedding_layer = tf.layers.Dense(args.embedding_size, activation=None, name="categories_embedding")
            #embeded = embedding_layer(tf.one_hot(self.images,args.number_of_categories))
            #next_layer = embeded
            next_layer = tf.one_hot(self.images,args.number_of_categories)
            print(next_layer)
            cnn = "CB-64-3-1-same,M-2-2,CB-64-3-1-same,M-2-2,F".split(sep=',')
            #cnn = "F"
            for layer in cnn:
                layer = layer.split('-')
                print(layer)
                if layer[0] == 'C':
                    next_layer = tf.layers.conv2d(next_layer, int(layer[1]), int(layer[2]), strides=(int(layer[3])), padding=layer[4],activation = tf.nn.relu)
                elif layer[0] == 'M':
                    next_layer = tf.layers.max_pooling2d(next_layer,(int(layer[1])), int(layer[2]))
                elif layer[0] == 'F':
                    next_layer = tf.layers.flatten(next_layer)
                elif layer[0] == 'R':
                    next_layer = tf.layers.dense(next_layer,int(layer[1]),activation=tf.nn.relu)
                elif layer[0] == 'CB':
                    next_layer = tf.layers.conv2d(next_layer, int(layer[1]), int(layer[2]), strides=(int(layer[3])), padding=layer[4])
                    next_layer = tf.contrib.layers.batch_norm(next_layer,)
                    next_layer = tf.nn.relu(next_layer)
                print(next_layer)

            dense_layer = tf.layers.dense(next_layer, 1024, activation=tf.nn.relu)
            
            output_layer = tf.layers.dense(dense_layer, args.labels_size, activation=tf.nn.sigmoid)
            self.output = output_layer
            ones = tf.ones((args.batch_size, args.labels_size), tf.float32)
            zeros = tf.zeros((args.batch_size, args.labels_size), tf.float32)
            
            self.predictions = tf.where(tf.greater(output_layer,0.5),ones,zeros)
            print(self.predictions)
            # Training
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits (labels=self.labels, logits=output_layer)          
            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.training = optimizer.minimize(self.loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32), axis=(0,1))
            print(self.accuracy)
            
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)
    
    def train(self, train, args):
        images, labels= train.next_batch(args.batch_size)
        loss, acc, pred ,_,_ = self.session.run([self.loss, self.accuracy,self.output, self.training, self.summaries["train"]],
                        {self.images: images, self.labels: labels})
        return loss, acc, pred

    
    def evaluate(self,name,dataset,args):
        images, labels = dataset.next_batch(args.batch_size)
        loss, acc,_= self.session.run([self.loss,self.accuracy, self.summaries[name]],
                        {self.images: images, self.labels: labels})
        return loss, acc
    
    
    
    
if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(0)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to pickled data")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--room_size", default=64, type=int, help="Size of the image representing room")
    parser.add_argument("--embedding_size", default=16, type=int, help="Size of embedding of the categories")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=40, type=int, help="Maximum number of threads to use.")
    
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself
    
    
    with open(os.path.join(args.folder,"train.pickle"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.folder,"val.pickle"), 'rb') as f:
        val_data = pickle.load(f)
    train = RoomClassDataset(train_data, args.room_size)
    val = RoomClassDataset(val_data, args.room_size)
    args.number_of_categories = train.get_number_of_categories()
    args.labels_size = len(train.room_cats)
    
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished(args.batch_size):
            acc,loss,pred = network.train(train,args)
        print("train loss: ", loss)
            #print("train preds: ", pred)
            #print("train acc: ", acc)
        acc,loss = network.evaluate("dev", val,args)
        print("dev loss: ", loss)
        print("dev acc: ", acc)
