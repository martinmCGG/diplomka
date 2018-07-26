#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from dataset.create_dataset import Dataset
import pickle
from prompt_toolkit import output

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
            self.sequences = tf.placeholder(tf.float32, [None, args.room_size, 7], name="sequences")
            self.labels = tf.placeholder(tf.bool, [None, 3], name="labels")

            next_layer = tf.layers.flatten(self.sequences)
            #TODO: Define Network
            for _ in range(args.hidden_count):
                next_layer = tf.layers.dense(next_layer, args.hidden_size ,activation=tf.nn.relu)
            
            output_layer = tf.layers.dense(next_layer, 3, activation=None)
            self.predictions = output_layer
            
            # Training
            loss = tf.losses.mean_squared_error(labels, self.predictions)
            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)
                
    def train(self, sequences, labels):
        acc,_,_ = self.session.run([self.accuracy, self.training, self.summaries["train"]], {self.sequences: sequences, self.labels: labels})
        return acc
    
    def evaluate(self, dataset, sequences, labels):
        acc, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.sequences: sequences, self.labels: labels})
        return acc
    
    
if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to pickled data")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--room_size", default=20, type=int, help="Number of items in room")
    parser.add_argument("--hidden_count", default=2, type=int, help="Number of hidden layers")
    parser.add_argument("--hidden_size", default=1024, type=int, help="Size of hidden layers")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=40, type=int, help="Maximum number of threads to use.")
    
    args = parser.parse_args()
    
    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself
    
    
    with open(os.path.join(args.folder,"train.pickle"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.folder,"val.pickle"), 'rb') as f:
        val_data = pickle.load(f)
    train = Dataset(train_data, args.room_size)
    val = Dataset(val_data, args.room_size)


    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            sequences, labels = train.next_batch(args.batch_size)
            network.train(sequences, labels)

        dev_sequences, dev_labels = val.all_data()
        print("{:.2f}".format(100 * network.evaluate("dev", dev_sequences, dev_labels)))
    
    