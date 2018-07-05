import tensorflow as tf
import numpy as np
from IPython import embed

class CNN_LSTM(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,num_hidden=128):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout

        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss

        #1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #2. CONVOLUTION LAYER + MAXPOOLING LAYER (per filter) ###############################
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # # CONVOLUTION LAYER
                # filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # conv = tf.nn.conv2d(self.embedded_chars_expanded, W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                # # NON-LINEARITY
                # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # # MAXPOOLING
                # pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                # pooled_outputs.append(pooled)

                filter_shape = [filter_size, embedding_size, 1, 28]
                W_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_1")
                b_1 = tf.Variable(tf.constant(0.1, shape=[28]), name="b_1")
                conv_1 = tf.nn.conv2d(self.embedded_chars_expanded, W_1, strides=[1, 1, 1, 1], padding="SAME", name="conv_1")
                # NON-LINEARITY
                # h_1 = tf.nn.relu(tf.nn.bias_add(conv_1, b_1), name="relu_1")
                # W_2 = tf.Variable(tf.truncated_normal([filter_size, embedding_size, 56, 18], stddev=0.1), name="W_2")
                # b_2 = tf.Variable(tf.constant(0.1, shape=[18]), name="b_2")
                # conv_2 = tf.nn.conv2d(h_1, W_2, strides=[1, 1, 1, 1], padding="SAME",
                #                       name="conv_2")
                h_2 = tf.nn.relu(tf.nn.bias_add(conv_1, b_1), name="relu_2")
                # MAXPOOLING
                pooled = tf.nn.max_pool(h_2, ksize=[1, 1, embedding_size, 1], strides=[1, 1, embedding_size, 1],
                                        padding='VALID', name="pool")
                # pooled_outputs.append(pooled)

        # COMBINING POOLED FEATURES
        h_pool = tf.squeeze(pooled, squeeze_dims=2)
        # h_pool = tf.squeeze(tf.concat(pooled_outputs, 3), squeeze_dims=2)
        # h_pool = tf.concat(pooled_outputs, 3)
        with tf.name_scope("dropout"):
             h_drop = tf.nn.dropout(h_pool, 0.2)
        self.h_pool_flat = tf.concat([self.embedded_chars, h_drop], 2)

        # 4. LSTM LAYER ######################################################################
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(cell, self.h_pool_flat, dtype=tf.float32)

        # COMBINING POOLED FEATURES
        # num_filters_total = num_filters * len(filter_sizes)
        # self.h_pool = tf.concat(pooled_outputs, 3)
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        #
        # #3. DROPOUT LAYER ###################################################################
        # with tf.name_scope("dropout"):
        #      self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # #4. LSTM LAYER ######################################################################
        # cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
        # self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        # val,state = tf.nn.dynamic_rnn(cell,self.h_drop_exp,dtype=tf.float32)
        #
        #embed()

        val2 = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val2, int(val2.get_shape()[0]) - 1)
        # last = state
        print('cat shapes', last.get_shape(), self.h_pool_flat.get_shape())
        out_weight = tf.Variable(tf.random_normal([128, num_classes]))
        out_bias = tf.Variable(tf.random_normal([num_classes]))

        with tf.name_scope("output"):
            #lstm_final_output = val[-1]
            #embed()
            self.scores = tf.nn.xw_plus_b(last, out_weight,out_bias, name="scores")
            # self.predictions = tf.nn.softmax(self.scores, name="predictions")
            self.predictions = tf.cast((tf.nn.sigmoid(self.scores) > 0.5), tf.float32)

        with tf.name_scope("loss"):
            self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),name="accuracy")

        print("(!) LOADED CNN-LSTM! :)")
        #embed()

