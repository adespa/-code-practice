#! /usr/bin/env python
import sys

#SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM   #OPTION 0
from lstm_cnn import LSTM_CNN   #OPTION 1
from cnn import CNN             #OPTION 2 (Model by: Danny Britz)
from lstm import LSTM           #OPTION 3
MODEL_TO_RUN = 1


import tensorflow as tf
import numpy as np
import os
import time
import datetime
import batchgen
from tensorflow.contrib import learn

from IPython import embed

# Parameters
# ==================================================

os.chdir('/Users/finogeeks/Documents/learn_datas/weibo_seg')
# Data loading params
dev_size = .10

# Model Hyperparameters
lstm_dim  = 128     #128
max_seq_legth = 80
filter_sizes = [3,4,5]  #3
num_filters = 32
dropout_prob = 0.5 #0.5
l2_reg_lambda = 0.0
use_glove = True #Do we use glove

# Training parameters
batch_size = 64
num_epochs = 10 #200
evaluate_every = 30 #100
checkpoint_every = 1000 #100
num_checkpoints = 0 #Checkpoints to store


# Misc Parameters
allow_soft_placement = True
log_device_placement = False



# Data Preparation
# ==================================================


filename = "../tweets.csv"
xfile = 'train_data/train_x.txt'
yfile = 'train_data/train_y.txt'
# test_xfile = 'train_data/model_test_x.txt'
# test_yfile = 'train_data/model_test_y.txt'
test_xfile = 'train_data/pos_x.txt'
test_yfile = 'train_data/pos_y.txt'

def loadfile(xfile, yfile):
    data_x = []
    data_y = []
    with open(xfile, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip()
            data_x.append(line)
    with open(yfile, 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            line = line.strip()
            data_y.append(int(line))

    return data_x,data_y

# Load data
print("Loading data...")
# x_text, y = batchgen.get_dataset(goodfile, badfile, 5000) #TODO: MAX LENGTH
x_text, y = loadfile(xfile, yfile)
nx_text, y_test = loadfile(test_xfile, test_yfile)
print('test:', len(nx_text), len(y_test), nx_text[0])

# Build vocabulary
max_document_length = 80

print('Tokenising...')
combined = batchgen.tokenizer(x_text)
test_combined = batchgen.tokenizer(nx_text)

if (not use_glove):
    print("Not using GloVe")
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
else:
    print('Training a Word2vec model...')
    embedding_dim = 200
    index_dict, word_vectors, combined = batchgen.word2vec_train(combined, maxlen=max_seq_legth)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_te, y_te = batchgen.get_data(index_dict, word_vectors, combined, y, vocab_dim=embedding_dim)
    print('train:', len(x_train), len(y_train))
    x_test = batchgen.get_test_dataset(test_combined, index_dict)
    print('the test embedding', len(x_test[0]),x_test[0])

    # W = tf.Variable(tf.constant(0.0, shape=[n_symbols, embedding_dim]),
    #                 trainable=True, name="W")
    # embedding_placeholder = tf.placeholder(tf.float32, [n_symbols, embedding_dim])
    # embedding_init = W.assign(embedding_placeholder)
    #
    # session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # sess = tf.Session(config=session_conf)
    # sess.run(embedding_init, feed_dict={embedding_placeholder: embedding_weights})


print("Vocabulary Size: {:d}".format(n_symbols))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

#embed()


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #embed()
        if (MODEL_TO_RUN == 0):
            model = CNN_LSTM(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embedding_dim,filter_sizes,num_filters,l2_reg_lambda)
        elif (MODEL_TO_RUN == 1):
            model = LSTM_CNN(max_seq_legth, 1, n_symbols, embedding_dim,filter_sizes,num_filters,l2_reg_lambda, weight=embedding_weights)
        elif (MODEL_TO_RUN == 2):
            model = CNN(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embedding_dim,filter_sizes,num_filters,l2_reg_lambda)
        elif (MODEL_TO_RUN == 3):
            model = LSTM(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embedding_dim)
        else:
            print("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
            exit()


        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        global_step = tf.Variable(1e-4, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(global_step)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        ckpt = tf.train.get_checkpoint_state('./runs/checkpoints/')
        if ckpt:
            print('loading model....')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        #TRAINING STEP
        def train_step(x_batch, y_batch,save=False):
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: dropout_prob
            }
            _, summaries, loss, accuracy, pred_ = sess.run(
                [train_op, train_summary_op, model.loss, model.accuracy, model.predictions],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(epoch, step, loss, accuracy))
            # print(pred_)
            if save:
                train_summary_writer.add_summary(summaries, step)

        #EVALUATE MODEL
        def dev_step(x_batch, y_batch, writer=None,save=False):
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: 0.5
            }
            summaries, loss, accuracy = sess.run(
                [dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                if writer:
                    writer.add_summary(summaries, step)

        #CREATE THE BATCHES GENERATOR
        batches = len(y_train) // batch_size
        
        #TRAIN FOR EACH BATCH
        for epoch in range(num_epochs):
            step = 0
            sess.run(global_step.assign(1e-4 * 0.9 ** epoch))
            for batch in range(batches):
                step += 1
                x_batch = x_train[batch*batch_size:(batch+1)*batch_size]
                y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]
                y_batch = np.expand_dims(y_batch, -1)
                train_step(x_batch, y_batch)
                # current_step = tf.train.global_step(sess, global_step)
                if step % evaluate_every == 0:
                    print("\nEvaluation:")
                    # tt = np.random.randint(0,3,1)[0]
                    tt = 0
                    test_xbatch = x_test[tt*4000:(tt+1)*4000]
                    test_ybatch = y_test[tt*4000:(tt+1)*4000]
                    dev_step(test_xbatch, np.expand_dims(test_ybatch, -1), writer=False)
                    print("")
                # if current_step % checkpoint_every == 0:
                #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #     print("Saved model checkpoint to {}\n".format(path))
            path = saver.save(sess, checkpoint_prefix, global_step=step)
            print("Saved model checkpoint to {}\n".format(path))
            dev_step(x_test[:5000], np.expand_dims(y_test[:5000],-1), writer=dev_summary_writer)
