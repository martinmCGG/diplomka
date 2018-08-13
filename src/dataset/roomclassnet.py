import numpy as np
import tensorflow as tf
from create_dataset import Dataset, RoomClassDataset
from parse_dataset import Data, Room, Model
import pickle
import math
from tensorflow.contrib.tensorboard.plugins import projector

class Network:
    def __init__(self, threads, seed=0):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))
        
    def construct(self, args):
        FURNITURE_CATS = args.number_of_categories
        with self.session.graph.as_default():        
            
            self.images = tf.placeholder(tf.int32, [None, args.room_size, args.room_size], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.isTraining = tf.placeholder(tf.bool, name = "isTraining")
            
            embedded_ids = self._construct_embeddings(FURNITURE_CATS,args.embedding_size, self.images)
            #network_string = "CB-32-3-1-same,M-2-2,CB-32-3-1-same,M-2-2,F,R-512,D"
            #cnn = "CB-64-3-1-same,M-2-2,CB-64-3-1-same,M-2-2,F,R-1024"
            #cnn = "F,R-256" 0.78
            network_string= "F,R-512,D"
            #cnn = "CB-32-3-2-same,M-2-2,CB-32-3-2-same,M-2-2,F,R-128"
            
            output_layer = self._construct_network(network_string, embedded_ids)
            self.predictions = tf.argmax(output_layer, axis=1)
            
            # Training
            self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, args.labels_size), output_layer)         
            self.global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            
            
            self.training = optimizer.minimize(self.loss, global_step=self.global_step, name="training")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            
            
            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            #self._write_summaries(args.logdir)
            self._write_summaries2(args.logdir, args.metadata_path)


    
    
    def _construct_embeddings(self, number_of_categories, embedding_size, data):
        self.embedding_var = tf.get_variable('embeddings', [number_of_categories, embedding_size])
        embedded_ids = tf.gather(self.embedding_var, data)    
        return embedded_ids
    
    def _construct_network(self, network_string, input):
        next_layer = input
        print(next_layer)
        cnn = network_string.split(sep=',')
        for layer in cnn:
            layer = layer.split('-')
            if layer[0] == 'C':
                next_layer = tf.layers.conv2d(next_layer, int(layer[1]), int(layer[2]), strides=(int(layer[3])), padding=layer[4],activation = tf.nn.relu)
            elif layer[0] == 'M':
                next_layer = tf.layers.max_pooling2d(next_layer,(int(layer[1])), int(layer[2]))
            elif layer[0] == 'F':
                next_layer = tf.layers.flatten(next_layer)
            elif layer[0] == 'R':
                next_layer = tf.layers.dense(next_layer,int(layer[1]),activation=tf.nn.relu)
            elif layer[0] == 'D':
                next_layer = tf.layers.dropout(next_layer, training=self.isTraining)
            elif layer[0] == 'CB':
                next_layer = tf.layers.conv2d(next_layer, int(layer[1]), int(layer[2]), strides=(int(layer[3])), padding=layer[4])
                next_layer = tf.contrib.layers.batch_norm(next_layer,)
                next_layer = tf.nn.relu(next_layer)
            print(next_layer)
        
        output_layer = tf.layers.dense(next_layer, args.labels_size, activation=None)
        return output_layer
    
    def _write_summaries(self, logdir):
        self.summaries = {}
    
        summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10 * 1000)
        
        with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
            self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                       tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            for dataset in ["dev", "test"]:
                self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.loss),
                                           tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]
        with summary_writer.as_default():
            tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)
          
    
    def _write_summaries2(self, logdir, metadata_path):
        
        self.accuracy_value = tf.placeholder(tf.float32, shape=())
        self.loss_value = tf.placeholder(tf.float32, shape=())
        
        self.summary_writer_train = tf.summary.FileWriter(os.path.join(logdir,"train"), graph=tf.get_default_graph())
        self.summary_writer_dev = tf.summary.FileWriter(os.path.join(logdir,"dev"))
        
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embedding_var.name
        embedding.metadata_path = metadata_path

        projector.visualize_embeddings(self.summary_writer_train, config)
        
        loss_summary = tf.summary.scalar("loss", self.loss_value)
        accuracy_summary = tf.summary.scalar("accuracy", self.accuracy_value)
        self.merged_summary_op = tf.summary.merge([loss_summary,accuracy_summary])
    
    def train(self, train, args):
        images, labels= train.next_batch(args.batch_size)
        loss, acc, pred ,_ = self.session.run([self.loss, self.accuracy, self.predictions, self.training],
                        {self.images: images, self.labels: labels, self.isTraining : True})
        return loss, acc, pred

    
    def evaluate(self,name,dataset,args):
        images, labels = dataset.next_batch(args.batch_size)
        loss, acc= self.session.run([self.loss, self.accuracy],
                        {self.images: images, self.labels: labels, self.isTraining : False})
        
        #self.summary_writer_dev.add_summary(summary, self.global_step)
        return loss, acc
    
    def summarize(self, accuracy, loss, train, step):
        
        summary = self.session.run([self.merged_summary_op],{self.accuracy_value:accuracy, self.loss_value:loss})
        if train:
            self.summary_writer_train.add_summary(summary[0], step)
        else:
            self.summary_writer_dev.add_summary(summary[0], step)
            
        saver = tf.train.Saver([self.embedding_var])
        saver.save(self.session, os.path.join(args.logdir, "model.ckpt"), step)
            
            
if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(1)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to pickled data")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--room_size", default=64, type=int, help="Size of the image representing room")
    parser.add_argument("--embedding_size", default=8, type=int, help="Size of embedding of the categories")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=40, type=int, help="Maximum number of threads to use.")
    
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself
    args.metadata_path = 'D:\\workspace\\diplomka\\METADATA.tsv'
    print(args.metadata_path)
    
    
    with open(os.path.join(args.folder,"train.pickle"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.folder,"val.pickle"), 'rb') as f:
        val_data = pickle.load(f)
    train = RoomClassDataset(train_data, args.room_size)
    val = RoomClassDataset(val_data, args.room_size)
    args.number_of_categories = train.get_number_of_categories()
    args.labels_size = len(train.room_cats)
    
    print(train.model_cats)
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    step = 0
    # Train
    for i in range(args.epochs):
        epoch_accuracy = 0
        epoch_loss = 0
        while not train.epoch_finished(args.batch_size):
            loss,acc,pred = network.train(train, args)
            epoch_accuracy += acc
            epoch_loss += loss
            step +=1
        epoch_accuracy /= step/(i+1)
        epoch_loss /= step/(i+1)
        
        network.summarize(epoch_accuracy, epoch_loss, True, step)
            
        print("train loss: ", epoch_loss)
        print("train acc: ", epoch_accuracy)
        
        val_step = 0
        val_loss = 0
        val_accuracy = 0
        while not val.epoch_finished(args.batch_size):
            loss, acc = network.evaluate("dev", val, args)
            val_loss += loss
            val_accuracy += acc
            val_step+=1
        
        network.summarize(val_accuracy/val_step, val_loss/val_step, False, step)
        print("dev loss: ", val_loss/val_step)
        print("dev acc: ", val_accuracy/val_step)
