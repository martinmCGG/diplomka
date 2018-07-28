import numpy as np
import tensorflow as tf
from create_dataset import Dataset
from parse_dataset import Data, Room, Model
import pickle
import math

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))
    
    def construct(self, args):
        with self.session.graph.as_default():
            
            # Inputs
            self.categories = tf.placeholder(tf.int32, [None, args.room_size], name="categories")
            self.sequences = tf.placeholder(tf.float32, [None, args.room_size, 6], name="sequences")
            self.labels = tf.placeholder(tf.float32, [None, 3], name="labels")
            self.labels_categories = tf.placeholder(tf.int32, [None], name="labels_categories")
            self.sequence_lengths = tf.placeholder(tf.int32, [None], name="lengths")
            
            #create and run embedding layers
            embedding_layer = tf.layers.Dense(args.embedding_size, activation=None, name="categories_embedding")
            embeded = embedding_layer(tf.one_hot(self.categories,args.number_of_categories))
            
            inputs = tf.concat((tf.squeeze(embeded), self.sequences), axis=2, name="input")
            print(inputs)
            inputs = tf.reshape(inputs, [-1,inputs.shape[1],args.embedding_size + 6])
            #create rnn cell and run recurent network
            cell = tf.nn.rnn_cell.LSTMCell(args.dims)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.sequence_lengths, dtype=tf.float32, scope ="rnnrun")
            
            #embed labeled item category
            embeded_label = embedding_layer(tf.one_hot(self.labels_categories,args.number_of_categories))
            
            proccesed_input = tf.concat((embeded_label,state[1]), axis = 1)
            print(state[1])
            print(proccesed_input)
            dense_layer = tf.layers.dense(proccesed_input, 512, activation=tf.nn.relu)
            
            output_layer = tf.layers.dense(dense_layer, 3, activation=None)
            self.predictions = output_layer
            print(self.predictions)
            print(self.labels)
            
            # Training
            self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.training = optimizer.minimize(self.loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.abs(self.labels - self.predictions))
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
                
    def train(self, train,args):
        sequences, labels, categories, labels_categories, lengths = train.next_batch(args.batch_size)
        loss,acc,_,_ = self.session.run([self.loss,self.accuracy, self.training, self.summaries["train"]],
                                         {self.sequences: sequences, self.labels: labels,self.categories:categories,self.labels_categories:labels_categories,self.sequence_lengths:lengths})
        return acc,loss
    
    def evaluate(self, name, dataset):
        sequences, labels, categories, labels_categories, lengths = dataset.all_data()
        loss,acc, _ = self.session.run([self.loss,self.accuracy, self.summaries[name]],
                                       {self.sequences: sequences, self.labels: labels,self.categories:categories,self.labels_categories:labels_categories,self.sequence_lengths:lengths})
        return acc,loss




if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to pickled data")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--room_size", default=30, type=int, help="Number of items in room")
    parser.add_argument("--embedding_size", default=8, type=int, help="Size of embedding of the categories")
    parser.add_argument("--dims", default=50, type=int, help="Number of hidden layers")
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
    train = Dataset(train_data, args.room_size)
    val = Dataset(val_data, args.room_size)
    args.number_of_categories = train.get_number_of_categories()
    print(args.number_of_categories)
    
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            acc,loss = network.train(train,args)
            print("train loss: ", loss)
            print("train acc: ", acc)

        acc,loss = network.evaluate("dev", val)
        print("dev loss: ", loss)
        print("dev acc: ", acc)