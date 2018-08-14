import numpy as np
import tensorflow as tf
from create_dataset import Dataset, ConvDataset
from parse_dataset import Data, Room, Model
import pickle
import math
from tensorflow.contrib.tensorboard.plugins import projector
from multiprocessing import reduction

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))
        
    def construct(self, args):
        FURNITURE_CATS = args.number_of_categories
        with self.session.graph.as_default():        
            
            self.images = tf.placeholder(tf.int32, [None, args.room_size,args.room_size], name="images")
            self.labels = tf.placeholder(tf.int32, [None,2], name="labels")
            self.labels_categories = tf.placeholder(tf.int32, [None], name="labels_categories")
            self.isTraining = tf.placeholder(tf.bool, name = "isTraining")
            
            embedded_ids = self._construct_embeddings(FURNITURE_CATS,args.embedding_size, self.images)
            network_string = "CB-64-3-1-same,CB-64-2-1-same,M-2-2,F"
            
            next_layer = self._construct_network(network_string, embedded_ids)
            
            embeded_label = tf.gather(self.embedding_var, self.labels_categories)
            
            next_layer = tf.concat((embeded_label, next_layer), axis = 1)

            next_layer = tf.layers.dense(next_layer, 1024, activation=tf.nn.relu)
            
            
            output_layer = tf.layers.dense(next_layer, 2, activation=None)
            self.predictions = output_layer
            
            # Training
            self.loss = tf.losses.mean_squared_error(tf.cast(self.labels, tf.float32), self.predictions, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            #self.accuracy = 0
            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.training = optimizer.minimize(self.loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
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
        return next_layer 
    
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
        #accuracy_summary = tf.summary.scalar("accuracy", self.accuracy_value)
        self.merged_summary_op = tf.summary.merge([loss_summary])
        
        self.saver = tf.train.Saver([self.embedding_var], max_to_keep=5, keep_checkpoint_every_n_hours = 2) 
    
    def train(self, train, args):
        images, labels,labels_categories = train.next_batch(args.batch_size)
        loss, _= self.session.run([self.loss, self.training],
                        {self.images: images, self.labels: labels,self.labels_categories:labels_categories,self.isTraining : True})
        return loss

    
    def evaluate(self,name,dataset,args):
        images, labels,labels_categories = dataset.next_batch(args.batch_size)
        loss = self.session.run([self.loss],
                        {self.images: images, self.labels: labels,self.labels_categories:labels_categories,self.isTraining : False})
        return loss[0]
    
    def summarize(self, loss, train, step):
        summary = self.session.run([self.merged_summary_op],{self.loss_value:loss})
        if train:
            self.summary_writer_train.add_summary(summary[0], step)
        else:
            self.summary_writer_dev.add_summary(summary[0], step)
            
        self.saver.save(self.session, os.path.join(args.logdir, "model.ckpt"), step)
    
if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to pickled data")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--room_size", default=64, type=int, help="Number of items in room")
    parser.add_argument("--embedding_size", default=8, type=int, help="Size of embedding of the categories")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=40, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--log_frequency", default=100, type=int, help="Frequency of training logging")
    
    args = parser.parse_args()
    
    # Create logdir name
    args.logdir = "logs/{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself
    args.metadata_path = 'D:\\workspace\\diplomka\\METADATA.tsv'
    
    with open(os.path.join(args.folder,"strain.pickle"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.folder,"sval.pickle"), 'rb') as f:
        val_data = pickle.load(f)
    train = ConvDataset(train_data, args.room_size)
    val = ConvDataset(val_data, args.room_size)
    args.number_of_categories = train.get_number_of_categories()
    
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)


    print("Starting to run training...")
    step = 0
    epoch_loss = 0
    # Train
    for i in range(args.epochs):
        #epoch_accuracy = 0
        while not train.epoch_finished(args.batch_size):
            loss = network.train(train, args)
            epoch_loss += math.sqrt(loss)
            step +=1
            if step % args.log_frequency == 0:
                epoch_loss /= args.log_frequency
                print("train loss: ", epoch_loss)
                network.summarize(epoch_loss, True, step)
                epoch_loss = 0
            
        val_step = 0
        val_loss = 0
        #val_accuracy = 0
        while not val.epoch_finished(args.batch_size):
            loss  = network.evaluate("dev", val, args)
            val_loss += math.sqrt(loss)
            val_step+=1
                
        
        network.summarize( val_loss/val_step, False, step)
        print("dev loss: ", val_loss/val_step)
        #print("dev acc: ", val_accuracy/val_step)