import tensorflow as tf
import numpy as np
import os
from Logger import Logger
from seq_rnn_model import SequenceRNNModel
import model_data
import csv
from config import get_config

config = get_config()

def train():
    data =  model_data.read_data(config.data, config)
    test_data = data.test
    seq_rnn_model = SequenceRNNModel(config.n_input_fc, config.num_views, config.n_hidden, config.decoder_embedding_size, config.num_classes+1, config.n_hidden,
                                     learning_rate=config.learning_rate,
                                     keep_prob=config.keep_prob,
                                     batch_size=config.batch_size,
                                     is_training=True,
                                     use_lstm=config.use_lstm,
                                     use_attention=config.use_attention,
                                     use_embedding=config.use_embedding,
                                     num_heads=config.num_heads)
                                     #init_decoder_embedding=model_data.read_class_yes_embedding(config.log_dir))
                                     
    seq_rnn_model_test = SequenceRNNModel(config.n_input_fc, config.num_views, config.n_hidden, config.decoder_embedding_size, config.num_classes+1, config.n_hidden,
                                 batch_size=test_data.size(),
                                 is_training=False,
                                 use_lstm=config.use_lstm,
                                 use_attention=config.use_attention,
                                 use_embedding=config.use_embedding,
                                 num_heads=config.num_heads)

    seq_rnn_model_test.build_model("eval")
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    
    with tf.Session(config=tf_config) as sess:
        seq_rnn_model.build_model("train")
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        start_epoch = 0
        WEIGHTS = config.weights
        if WEIGHTS!=-1:
            ld = config.log_dir
            start_epoch = WEIGHTS + 1
            saver.restore(sess, get_modelpath(WEIGHTS))
            ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),
                             os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = WEIGHTS)
            LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)),
                               os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = WEIGHTS)
        
        accs = []
        losses = []
        for epoch in xrange(start_epoch, config.max_epoch + start_epoch + 1):
            batch = 1
            while batch * config.batch_size <= data.train.size():
                batch_encoder_inputs, batch_decoder_inputs = data.train.next_batch(config.batch_size)
                target_labels = get_target_labels(batch_decoder_inputs)
                batch_encoder_inputs = batch_encoder_inputs.reshape((config.batch_size, config.num_views, config.n_input_fc))
                batch_encoder_inputs, batch_decoder_inputs, batch_target_weights = seq_rnn_model.get_batch(batch_encoder_inputs, batch_decoder_inputs, batch_size=config.batch_size)
                loss, logits = seq_rnn_model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_target_weights,forward_only=False)
                predict_labels = seq_rnn_model.predict(logits)
                acc = accuracy(predict_labels, target_labels)
                accs.append(acc)
                losses.append(loss)
                
                if batch % max(config.train_log_frq/config.batch_size,1) == 0:
                    loss = np.mean(losses)
                    acc = np.mean(accs)
                    LOSS_LOGGER.log(loss, epoch, "train_loss")
                    ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    print("epoch %d batch %d: loss=%f acc=%f" %(epoch, batch, loss, acc))
                    accs = []
                    losses = []
                batch += 1

            if epoch % config.save_period == 0:
                saver.save(sess, get_modelpath(epoch))
            
            weights = seq_rnn_model.get_weights(sess)
            eval_during_training(weights, seq_rnn_model_test, epoch)
    
            ACC_LOGGER.save(config.log_dir)
            LOSS_LOGGER.save(config.log_dir)
            ACC_LOGGER.plot(dest=config.log_dir)
            LOSS_LOGGER.plot(dest=config.log_dir)


def _test(data, seq_rnn_model, sess):
    
    test_encoder_inputs, test_decoder_inputs = data.next_batch(data.size(), shuffle=False)
    target_labels = get_target_labels(test_decoder_inputs)
    
    test_encoder_inputs = test_encoder_inputs.reshape((-1, config.num_views, config.n_input_fc))
    test_encoder_inputs, test_decoder_inputs, test_target_weights = seq_rnn_model.get_batch(test_encoder_inputs,
                                                                                            test_decoder_inputs,batch_size=data.size())
    
    logits, loss = seq_rnn_model.step(sess, test_encoder_inputs, test_decoder_inputs, test_target_weights, forward_only=True)  # don't do optimize

    predict_labels = seq_rnn_model.predict(logits, all_min_no=False)

    acc = accuracy(predict_labels, target_labels)
    return acc, loss, predict_labels, target_labels

def eval_during_training(weights, model, epoch):
    data = model_data.read_data(config.data, config, read_train=False)
    data = data.test
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.assign_weights(sess, weights, "eval")
        acc,loss, _, _ = _test(data, model, sess)
    print("evaluation, acc=%f" %(acc[0]))
    LOSS_LOGGER.log(loss, epoch,"eval_loss")
    ACC_LOGGER.log(acc[0],epoch, "eval_accuracy")
    
    
def eval_alone():
    data = model_data.read_data(config.data, config, read_train=False)
    data = data.test
    seq_rnn_model = SequenceRNNModel(config.n_input_fc, config.num_views, config.n_hidden, config.decoder_embedding_size, config.num_classes+1, config.n_hidden,
                                     batch_size=data.size(),
                                     is_training=False,
                                     use_lstm=config.use_lstm,
                                     use_attention=config.use_attention,
                                     use_embedding=config.use_embedding, num_heads=config.num_heads)
    seq_rnn_model.build_model("train")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
        saver = tf.train.Saver()
        saver.restore(sess, get_modelpath(config.weights))
        acc, loss, predictions, labels = _test(data, seq_rnn_model, sess)
    print("model:%s, acc_instance=%f, acc_class=%f" % ("Model", acc[0], acc[1]))
    
    predictions = [x-1 for x in predictions]  
    labels = [x-1 for x in labels]
    
    import Evaluation_tools as et
    eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
    et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
    et.make_matrix(config.data, eval_file, config.log_dir)    
        

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
        with open("class_acc.csv", 'w') as f:
            w = csv.writer(f)
            for k in acc_classes_map:
                w.writerow([k, acc_classes_map[k]])
        return  [np.mean(np.equal(predict, target)), np.mean(np.array(acc_classes))]

def get_modelpath(epoch):
    return os.path.join(config.log_dir, config.snapshot_prefix + str(epoch))

def main(argv):
    
    if config.test:
        eval_alone()
    else:
        train()

if __name__ == '__main__':
    LOSS_LOGGER = Logger("{}_loss".format(config.name))
    ACC_LOGGER = Logger("{}_acc".format(config.name))
    tf.app.run(main)
    


