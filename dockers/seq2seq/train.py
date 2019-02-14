import tensorflow as tf
import numpy as np
import os
from Logger import Logger
from seq_rnn_model import SequenceRNNModel
import model_data
import csv
from sys import argv
import Evaluation_tools as et


# data path parameter
tf.flags.DEFINE_string('log_dir', '', 'file dir for saving features and labels')
tf.flags.DEFINE_string('data', '', 'path to data')
tf.flags.DEFINE_string('weights', '', 'number of model to finetune or test')


tf.flags.DEFINE_string('checkpoint_path', './logs', 'trained model checkpoint')
tf.flags.DEFINE_string('test_acc_file', 'seq_acc.csv', 'test acc file')

# model parameter
tf.flags.DEFINE_boolean("use_embedding", True, "whether use embedding")
tf.flags.DEFINE_boolean("use_attention", True, "whether use attention")

tf.flags.DEFINE_integer("training_epoches", 50, "total train epoches")
tf.flags.DEFINE_integer("save_epoches", 5, "epoches can save")
tf.flags.DEFINE_integer("n_views", 12, "number of views for each model")
tf.flags.DEFINE_integer("n_input_fc", 4096, "size of input feature")
tf.flags.DEFINE_integer("decoder_embedding_size", 256, "decoder embedding size")
tf.flags.DEFINE_integer("n_classes", 40, "total number of classes to be classified")
tf.flags.DEFINE_integer("n_hidden", 128, "hidden of rnn cell")
tf.flags.DEFINE_float("keep_prob", 1.0, "kepp prob of rnn cell")
tf.flags.DEFINE_boolean("use_lstm", False, "use lstm or gru cell")

# attention parameter
tf.flags.DEFINE_integer("num_heads", 1, "Number of attention heads that read from attention_states")

# training parameter
tf.flags.DEFINE_boolean('train', True, 'train mode')
tf.flags.DEFINE_integer("batch_size", 32, "training batch size")
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.flags.DEFINE_integer("n_max_keep_model", 1000, "max number to save model")

FLAGS = tf.flags.FLAGS

def main(argv):
    if FLAGS.train:
        train(FLAGS.weights)
    else:
        eval_alone(FLAGS.weights)


def train(weights):
    LOG_DIR = FLAGS.log_dir
    data =  model_data.read_data(FLAGS.data, n_views=FLAGS.n_views)
    test_data = data.test
    seq_rnn_model = SequenceRNNModel(FLAGS.n_input_fc, FLAGS.n_views, FLAGS.n_hidden, FLAGS.decoder_embedding_size, FLAGS.n_classes+1, FLAGS.n_hidden,
                                     learning_rate=FLAGS.learning_rate,
                                     keep_prob=FLAGS.keep_prob,
                                     batch_size=FLAGS.batch_size,
                                     is_training=True,
                                     use_lstm=FLAGS.use_lstm,
                                     use_attention=FLAGS.use_attention,
                                     use_embedding=FLAGS.use_embedding,
                                     num_heads=FLAGS.num_heads)
                                     #init_decoder_embedding=model_data.read_class_yes_embedding(FLAGS.log_dir))
                                     
    seq_rnn_model_test = SequenceRNNModel(FLAGS.n_input_fc, FLAGS.n_views, FLAGS.n_hidden, FLAGS.decoder_embedding_size, FLAGS.n_classes+1, FLAGS.n_hidden,
                                 batch_size=test_data.size(),
                                 is_training=False,
                                 use_lstm=FLAGS.use_lstm,
                                 use_attention=FLAGS.use_attention,
                                 use_embedding=FLAGS.use_embedding,
                                 num_heads=FLAGS.num_heads)

    seq_rnn_model_test.build_model("eval")
    
    config = tf.ConfigProto()
    
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    
    
    #if not os.path.exists(get_modelpath()):
    #    os.makedirs(get_modelpath())
    with tf.Session(config=config) as sess:
        seq_rnn_model.build_model("train")
        saver = tf.train.Saver(max_to_keep=FLAGS.n_max_keep_model)
        init = tf.global_variables_initializer()
        sess.run(init)
        startepoch = 0
        
        if weights:
            weights = int(weights)
            w = os.path.join(args.log_dir, "model.ckpt-{}".format(weights))
            saver.restore(sess, w)
            startepoch = weights + 1
            ACC_LOGGER.load((os.path.join(FLAGS.log_dir,"seq2seq_acc_train_accuracy.csv"),os.path.join(FLAGS.log_dir,"seq2seq_acc_eval_accuracy.csv")), epoch=weights)
            LOSS_LOGGER.load((os.path.join(FLAGS.log_dir,"seq2seq_loss_train_loss.csv"), os.path.join(FLAGS.log_dir,'seq2seq_loss_eval_loss.csv')), epoch=weights)
            
        epoch = startepoch
        
        accs = []
        losses = []
        while epoch <= FLAGS.training_epoches + startepoch:
            batch = 1
            
            while batch * FLAGS.batch_size <= data.train.size():
                batch_encoder_inputs, batch_decoder_inputs = data.train.next_batch(FLAGS.batch_size)
                target_labels = get_target_labels(batch_decoder_inputs)
                batch_encoder_inputs = batch_encoder_inputs.reshape((FLAGS.batch_size, FLAGS.n_views, FLAGS.n_input_fc))
                batch_encoder_inputs, batch_decoder_inputs, batch_target_weights = seq_rnn_model.get_batch(batch_encoder_inputs, batch_decoder_inputs, batch_size=FLAGS.batch_size)
                loss, logits = seq_rnn_model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_target_weights,forward_only=False)
                predict_labels = seq_rnn_model.predict(logits)
                acc = accuracy(predict_labels, target_labels)
                accs.append(acc)
                losses.append(loss)
                
                if batch%10 == 0:
                    loss = np.mean(losses)
                    acc = np.mean(accs)
                    LOSS_LOGGER.log(loss, epoch, "train_loss")
                    ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    print("epoch %d batch %d: loss=%f" %(epoch, batch, loss))
                    accs = []
                    losses = []
                batch += 1
            # if epoch % display_epoch == 0:
            #     print("epoch %d:display" %(epoch))
            if epoch % FLAGS.save_epoches == 0 and epoch>0:
                saver.save(sess, get_modelpath(epoch))
            #     # do test using test dataset
            
            weights = seq_rnn_model.get_weights(sess)
            eval_during_training(weights, seq_rnn_model_test, epoch)
            epoch += 1
    
            ACC_LOGGER.save(LOG_DIR)
            LOSS_LOGGER.save(LOG_DIR)
            ACC_LOGGER.plot(dest=LOG_DIR)
            LOSS_LOGGER.plot(dest=LOG_DIR)


def _test(data, seq_rnn_model, sess):
    
    test_encoder_inputs, test_decoder_inputs = data.next_batch(data.size(), shuffle=False)
    target_labels = get_target_labels(test_decoder_inputs)
    
    test_encoder_inputs = test_encoder_inputs.reshape((-1, FLAGS.n_views, FLAGS.n_input_fc))
    test_encoder_inputs, test_decoder_inputs, test_target_weights = seq_rnn_model.get_batch(test_encoder_inputs,
                                                                                            test_decoder_inputs,batch_size=data.size())
    
    logits, loss = seq_rnn_model.step(sess, test_encoder_inputs, test_decoder_inputs, test_target_weights, forward_only=True)  # don't do optimize

    predict_labels = seq_rnn_model.predict(logits, all_min_no=False)

    acc = accuracy(predict_labels, target_labels)
    return acc, loss, predict_labels, target_labels

def eval_during_training(weights, model, epoch):
    data = model_data.read_data(FLAGS.data, n_views=FLAGS.n_views, read_train=False)
    data = data.test
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.assign_weights(sess, weights, "eval")
        acc,loss, _, _ = _test(data, model, sess)
    print("evaluation, acc=%f" %(acc[0]))
    LOSS_LOGGER.log(loss, epoch,"eval_loss")
    ACC_LOGGER.log(acc[0],epoch, "eval_accuracy")
    
    
def eval_alone(weights):
    data = model_data.read_data(FLAGS.data, n_views=FLAGS.n_views, read_train=False)
    data = data.test
    seq_rnn_model = SequenceRNNModel(FLAGS.n_input_fc, FLAGS.n_views, FLAGS.n_hidden, FLAGS.decoder_embedding_size, FLAGS.n_classes+1, FLAGS.n_hidden,
                                     batch_size=data.size(),
                                     is_training=False,
                                     use_lstm=FLAGS.use_lstm,
                                     use_attention=FLAGS.use_attention,
                                     use_embedding=FLAGS.use_embedding, num_heads=FLAGS.num_heads)
    seq_rnn_model.build_model("train")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
        saver = tf.train.Saver()
        weights = int(weights)
        w = os.path.join(FLAGS.log_dir, "model.ckpt-{}".format(weights))
        saver.restore(sess, w)    
        acc, loss, predictions, labels = _test(data, seq_rnn_model, sess)
    print("model:%s, acc_instance=%f, acc_class=%f" % ("Model", acc[0], acc[1]))
    
    predictions = [x-1 for x in predictions]  
    labels = [x-1 for x in labels]
    eval_file = os.path.join(FLAGS.log_dir, 'seq2seq.txt')
    et.write_eval_file(FLAGS.data, eval_file, predictions , labels , 'SEQ2SEQ')
    et.make_matrix(FLAGS.data, eval_file, FLAGS.log_dir)
        

def get_target_labels(seq_labels):
    target_labels = []
    for i in range(np.shape(seq_labels)[0]): #loop batch_size
        for j in range(np.shape(seq_labels)[1]): #loop label
            if seq_labels[i][j] % 2 == 1:
                target_labels.append((seq_labels[i][j]+1)/2)
                break
    return target_labels

def accuracy(predict, target, mode="average_class"):
    predict, target = np.array(predict), np.array(target)
    if mode == "average_instance":
        return np.mean(np.equal(predict, target))
    elif mode == "average_class":
        target_classes = np.unique(target)
        acc_classes = []
        acc_classes_map = {}
        for class_id in target_classes:
            predict_at_class = predict[np.argwhere(target == class_id).reshape([-1])]
            acc_classes.append(np.mean(np.equal(predict_at_class, class_id)))
            acc_classes_map[class_id] = acc_classes[-1]
        #print("class accuracy:", acc_classes_map)
        with open("class_acc.csv", 'w') as f:
            w = csv.writer(f)
            for k in acc_classes_map:
                w.writerow([k, acc_classes_map[k]])
        return  [np.mean(np.equal(predict, target)), np.mean(np.array(acc_classes))]

def get_modelpath(epoch):
    return os.path.join(FLAGS.log_dir, "model.ckpt-{}".format(epoch))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", default=False, type=bool, help="train or test")
    parser.add_argument('--weights')
    
    parser.add_argument("--n_hidden", default=128, type=int, help="number of hidden neurons")
    parser.add_argument("--decoder_embedding_size", default=256, type=int, help="")
    parser.add_argument("--n_views", default=20, type=int, help="number of views") 
    parser.add_argument("--use_lstm", default=False, type=bool, help="use gru or lstm")
    parser.add_argument("--keep_prob", default=0.5, type=float, help="droupout rate") 
    parser.add_argument("--training_epoches", default=100, type=int, help="number of epochs to train")
    parser.add_argument('--save_epoches', default=10,type=int)
    parser.add_argument("--learning_rate", default=0.0002, type=float, help="learning rate") 
    parser.add_argument("--batch_size", default=32, type=int, help="number of epochs to train")
    parser.add_argument('--n_max_keep_model', default=50, type=int)

    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--train_feature_file')
    parser.add_argument('--train_label_file')
    parser.add_argument('--test_feature_file')
    parser.add_argument('--test_label_file')
    parser.add_argument('--save_seq_embeddingmvmodel_path')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--test_acc_file')
    
    parser.add_argument('--data', default='/data/converted', type=str)
    parser.add_argument('--log_dir',default='logs', type=str)
    args = parser.parse_args()
    
    LOSS_LOGGER = Logger("seq2seq_loss")
    ACC_LOGGER = Logger("seq2seq_acc")

    FLAGS.log_dir = args.log_dir
    FLAGS.data = args.data
    FLAGS.weights = args.weights
    FLAGS.train = not args.test
    
    tf.app.run(main,argv)
    


