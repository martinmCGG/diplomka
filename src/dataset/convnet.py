import numpy as np
import tensorflow as tf
from create_dataset import Dataset, ConvDataset
from parse_dataset import Data, Room, Model
import pickle
import math
from tensorflow.contrib.tensorboard.plugins import projector
from multiprocessing import reduction
from imager import save_batch_as_image

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
            
            self._define_placeholders(args)
            
            input = self._construct_embeddings(FURNITURE_CATS,args.embedding_size, self.images)
            if args.type_of_prediction == 'map':
                embedded_cats = tf.gather(self.embedding_var, self.labels_categories)
                input = tf.concat([input, embedded_cats],-1)
            
            network_string = args.network
            
            next_layer = self._construct_network(network_string, input)
            
            self._construct_ouput(args, next_layer)
            
            self._define_loss(args, self.output)
            
            #self.accuracy = 0
            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.training = optimizer.minimize(self.loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            
            #self._write_summaries(args)
            self._write_summaries_contrib(args.logdir)
            
    def _construct_embeddings(self, number_of_categories, embedding_size, data):
        self.embedding_var = tf.get_variable('embeddings', [number_of_categories, embedding_size])
        embedded_ids = tf.gather(self.embedding_var, data)    
        return embedded_ids
    
    def _define_placeholders(self, args):
        self.images = tf.placeholder(tf.int32, [None, args.room_size,args.room_size], name="images")
        self.isTraining = tf.placeholder(tf.bool, name = "isTraining")
        
        if args.type_of_prediction == 'coordinates':
            self.labels = tf.placeholder(tf.float32, [None,2], name="labels")
            self.labels_categories = tf.placeholder(tf.int32, [None], name="labels_categories")
            
        elif args.type_of_prediction == 'map':
            self.labels = tf.placeholder(tf.float32, [None, args.room_size, args.room_size], name="labels")
            self.labels_categories = tf.placeholder(tf.int32, [None, args.room_size, args.room_size], name="labels_categories")
            
    def _construct_ouput(self, args, input):
        if args.type_of_prediction == 'coordinates':   
            output_layer = tf.layers.dense(input, 2, activation=None)
            self.output = output_layer
            self.predictions = output_layer
        elif args.type_of_prediction == 'map':
            dense = tf.layers.dense(input,args.room_size*args.room_size*2, activation=None)
            print(dense)
            self.output = tf.reshape(dense,(-1,args.room_size,args.room_size,2))
            print(self.output)
            self.predictions = tf.argmax(self.output,axis=-1)
            
    def _define_loss(self,args, input):
        if args.type_of_prediction == 'coordinates':
            self.loss = tf.losses.mean_squared_error(tf.cast(self.labels, tf.float32), self.output, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        elif args.type_of_prediction == 'map':
            labels = tf.cast(self.labels, tf.int32)
            self.loss =tf.losses.sparse_softmax_cross_entropy(labels, self.output,)
    
    def _construct_network(self, network_string, input):
        next_layer = input
        print(next_layer)
        cnn = network_string.split(sep=',')
        for layer in cnn:
            layer = layer.split('-')
            if layer[0] == 'C':
                if len(layer) == 5:
                    if layer[4] == 'relu':
                        activation = tf.nn.relu
                    else:
                        activation = None
                else:
                    activation = tf.nn.relu
                next_layer = tf.layers.conv2d(next_layer, int(layer[1]), int(layer[2]), strides=int(layer[3]), padding='same',activation = activation)
            elif layer[0] == 'M':
                next_layer = tf.layers.max_pooling2d(next_layer,(int(layer[1])), int(layer[2]))
            elif layer[0] == 'F':
                next_layer = tf.layers.flatten(next_layer)
            elif layer[0] == 'R':
                next_layer = tf.layers.dense(next_layer,int(layer[1]),activation=tf.nn.relu)
            elif layer[0] == 'D':
                next_layer = tf.layers.dropout(next_layer, training=self.isTraining)
            elif layer[0] == 'CB':
                next_layer = tf.layers.conv2d(next_layer, int(layer[1]), int(layer[2]), strides=int(layer[3]), padding='same')
                next_layer = tf.contrib.layers.batch_norm(next_layer,)
                next_layer = tf.nn.relu(next_layer)
            elif layer[0] == 'CT':
                next_layer = tf.layers.conv2d_transpose(next_layer, int(layer[1]), int(layer[2]), strides=int(layer[3]), padding='same', activation=tf.nn.relu)
            elif layer[0] == 'E':
                embeded_label = tf.gather(self.embedding_var, self.labels_categories)
                next_layer = tf.concat((embeded_label, next_layer), axis = 1)
            print(next_layer)
        return next_layer 
    
    def _write_summaries(self, args):
        
        logdir = args.logdir
        metadata_path = args.metadata_path
        
        self.summary_writer_train = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embedding_var.name
        embedding.metadata_path = metadata_path

        projector.visualize_embeddings(self.summary_writer_train, config)

        self.saver = tf.train.Saver([self.embedding_var], max_to_keep=5, keep_checkpoint_every_n_hours = 2) 
        
    def _write_summaries_contrib(self, logdir):
        
        self.mean_train_loss, _ = tf.metrics.mean(self.loss, updates_collections=['train_updates'],name ='train_vars' )
        self.mean_train_iou, _ = tf.metrics.mean(self.compute_iou(self.labels, self.predictions), updates_collections=['train_updates'], name='train_vars')
        
        self.mean_dev_loss, _ = tf.metrics.mean(self.loss,updates_collections=['dev_updates'], name="dev_vars")
        self.mean_dev_iou, _ = tf.metrics.mean(self.compute_iou(self.labels, self.predictions), updates_collections=['dev_updates'], name='dev_vars')
    
        self.train_vars_initializer = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='train_vars'))
        self.dev_vars_initializer = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='dev_vars'))
        self.session.run(self.train_vars_initializer)
        self.session.run(self.dev_vars_initializer)

        self.summaries = {}
    
        summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10 * 1000)
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.mean_train_loss),
                                       tf.contrib.summary.scalar("train/iou", self.mean_train_iou),
                                       tf.contrib.summary.image("train/predictions", tf.cast(tf.expand_dims(self.predictions,-1),tf.float32), max_images=1),
                                       tf.contrib.summary.image("train/labels", tf.expand_dims(self.labels,-1),max_images=1)]
        
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            for dataset in ["dev", "test"]:
                self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.mean_dev_loss),
                                           tf.contrib.summary.scalar(dataset +"/iou", self.mean_dev_iou),
                                           tf.contrib.summary.image(dataset +"/predictions", tf.cast(tf.expand_dims(self.predictions,-1),tf.float32),max_images=1),
                                           tf.contrib.summary.image(dataset +"/labels", tf.expand_dims(self.labels,-1),max_images=1)]
        with summary_writer.as_default():
            tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)
    
    def train(self, train, args, step):
        updates = self.session.graph.get_collection('train_updates')
        while not train.epoch_finished(args.batch_size):
            step +=1
            images, labels, labels_categories = train.next_batch(args.batch_size)
            feed = {self.images: images, self.labels: labels,self.labels_categories:labels_categories,self.isTraining : True}
            if step % args.log_frequency != 0:
                
                self.session.run([self.training, updates],feed)
            else:
                self.session.run([self.training, updates, self.summaries['train']],feed)
                if step % (args.log_frequency*100) == 0:
                    #self.save(step)
                    pass
               
        return step
    
    def evaluate(self, name, dataset, args):
        updates = self.session.graph.get_collection('dev_updates')
        self.session.run([self.dev_vars_initializer])
        while not dev.epoch_finished(args.batch_size):
            
            is_last = dataset.is_last_batch(args.batch_size)
            
            images, labels, labels_categories = dataset.next_batch(args.batch_size)
            feed = {self.images: images, self.labels: labels, self.labels_categories:labels_categories,self.isTraining : False}
                
            if is_last:
                self.session.run([self.summaries[name], updates], feed)
            else:
                self.session.run([updates], feed)
    
    def predict(self,dataset,args, name):
        updates = self.session.graph.get_collection('dev_updates')
        self.session.run([self.dev_vars_initializer])
        #while not dev.epoch_finished(args.batch_size):
        
        images, labels, labels_categories = dataset.next_batch(args.batch_size)
        feed = {self.images: images, self.labels: labels, self.labels_categories:labels_categories,self.isTraining : False}
        output, predictions, _ = self.session.run([self.output, self.predictions, updates], feed)
        save_batch_as_image([images, labels, predictions], name)
                
    def save(self, step):
        self.saver.save(self.session, os.path.join(args.logdir, "model.ckpt"), step)
        
    def print_parameters_count(self):
        with self.session.graph.as_default():
            params_count = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
            print("Constructed a network with {} parameters".format(params_count))
    
    def compute_iou(self, labels, predictions):
        predictions = tf.cast(predictions, tf.float32)
        intersection = tf.reduce_sum(labels * predictions, axis=[1,2])
        iou = tf.reduce_mean(
        intersection / (1+tf.reduce_sum(labels, axis=[1,2]) + tf.reduce_sum(predictions, axis=[1,2]) - intersection)
        )
        return iou
    
if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(0)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to pickled data")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--room_size", default=32, type=int, help="Number of items in room")
    parser.add_argument("--embedding_size", default=8, type=int, help="Size of embedding of the categories")
    parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=40, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--log_frequency", default=200, type=int, help="Frequency of training logging")
    "coordinates, map"
    parser.add_argument("--type_of_prediction", default="map", type=str, help="Type of predicted")
    parser.add_argument("--network", default="--network CB-64-2-1,CB-64-2-1,M-2-2,F,R-1024,D", type=str, help="String defining the network")
    
    args = parser.parse_args()
    
    # Create logdir name
    args.logdir = "logs/{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself
    args.metadata_path = 'D:\\workspace\\diplomka\\METADATA.tsv'
    
    with open(os.path.join(args.folder,"train.pickle"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.folder,"val.pickle"), 'rb') as f:
        dev_data = pickle.load(f)
    
    train = ConvDataset(train_data, args.room_size, args.type_of_prediction)
    
    
    dev = ConvDataset(dev_data, args.room_size, args.type_of_prediction)
    args.number_of_categories = train.get_number_of_categories()
    
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)
    network.print_parameters_count()

    print("Starting to run training...")
    step = 0
    for i in range(args.epochs):
        print("*********** Epoch {} ***********".format(i+1))
        step += network.train(train, args, step)
        print("Evaluating on dev set...")
        network.evaluate("dev", dev, args)
        print("Saving Images...")
        network.predict(train, args, "train" + str(i))
        network.predict(dev, args, "dev" + str(i))
    print("-------------------------------END-------------------------------")
    
    