import tensorflow as tf
import numpy as np
# from IPython import embed

class LSTM_CNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.02, weight=None):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout

        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss

        #1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.cast(weight, tf.float32), name="W")
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            # test new model
            self.W_fa = tf.Variable(tf.random_uniform([vocab_size, 1], -1.0, 1.0), name="W_fa")
            self.fa = tf.nn.tanh(tf.nn.embedding_lookup(self.W_fa, self.input_x))
            self.fa = tf.squeeze(self.fa, squeeze_dims=2)
            # self.embedded_chars = tf.multiply(self.embedded_chars, self.fa)

            # self.W_fa = tf.Variable(tf.random_uniform([vocab_size, 10], -1.0, 1.0), name="W_fa")
            # self.fa = tf.nn.embedding_lookup(self.W_fa, self.input_x)
            # self.fa = tf.expand_dims(tf.pad(self.fa, [[0,0],[1,1],[0,0]]), -1)
            # fw1 = tf.Variable(tf.random_uniform([3, 10, 1, 10], -1.0, 1.0), name='fw1')
            # fb1 = tf.Variable(tf.constant(0.2, shape=[10]), name='fb1')
            # self.fa = tf.nn.conv2d(self.fa, fw1, strides=[1,1,1,1], padding='VALID')
            # self.fa = tf.nn.relu(self.fa + fb1)
            # fw2 = tf.Variable(tf.random_uniform([1, 1, 10, 1], -1.0, 1.0), name='fw2')
            # fb2 = tf.Variable(tf.constant(0.2, shape=[1]), name='fb2')
            # self.fa = tf.nn.conv2d(self.fa, fw2, strides=[1, 1, 1, 1], padding='VALID')
            # self.fa = tf.nn.sigmoid(self.fa + fb2)
            # self.fa = tf.squeeze(self.fa, squeeze_dims=3)

            # self.W_fa = tf.Variable(tf.random_uniform([vocab_size, 10], -1.0, 1.0), name="W_fa")
            # self.fa = tf.nn.embedding_lookup(self.W_fa, self.input_x)
            # self.fa = tf.expand_dims(tf.pad(self.fa, [[0, 0], [1, 1], [0, 0]]), -1)
            # fw1 = tf.Variable(tf.random_uniform([3, 10, 1, 1], -1.0, 1.0), name='fw1')
            # fb1 = tf.Variable(tf.constant(0.2, shape=[1]), name='fb1')
            # self.fa = tf.nn.conv2d(self.fa, fw1, strides=[1, 1, 1, 1], padding='VALID')
            # self.fa = tf.nn.tanh(self.fa + fb1)
            # self.fa = tf.squeeze(self.fa, squeeze_dims=3)
            # self.embedded_chars = tf.multiply(self.embedded_chars, self.fa)


            # fw2 = tf.Variable(tf.random_uniform([1, 1, 10, 1], -1.0, 1.0), name='fw2')
            # fb2 = tf.Variable(tf.constant(0.2, shape=[1]), name='fb2')
            # self.fa = tf.nn.conv2d(self.fa, fw2, strides=[1, 1, 1, 1], padding='VALID')
            # self.fa = tf.nn.sigmoid(self.fa + fb2)
            # self.fa = tf.squeeze(self.fa, squeeze_dims=3)
            # self.embedded_chars = tf.multiply(self.embedded_chars, self.fa)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        #2. LSTM LAYER ######################################################################
        self.lstm_cell = tf.contrib.rnn.LSTMCell(128,state_is_tuple=True)
        #self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        self.lstm_out,self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,self.embedded_chars,dtype=tf.float32)

        # test word embedding factor
        val2 = tf.transpose(self.lstm_out, [1, 0, 2])
        last = tf.gather(val2, int(val2.get_shape()[0]) - 1)
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(last, self.dropout_keep_prob)
        with tf.name_scope("output"):
            self.h_drop = tf.concat([self.h_drop, self.fa], 1)
            W = tf.get_variable(
                "W",
                shape=[208, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predictions = tf.cast((tf.nn.sigmoid(self.scores) > 0.5), tf.float32)

        #embed()
        # self.lstm_out = tf.multiply(self.lstm_out, self.fa)
        # self.lstm_out = tf.nn.dropout(self.lstm_out, self.dropout_keep_prob)
        # self.lstm_out_expanded = tf.expand_dims(self.lstm_out, -1)
        #
        # #2. CONVOLUTION LAYER + MAXPOOLING LAYER (per filter) ###############################
        # pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("conv-maxpool-%s" % filter_size):
        #         # CONVOLUTION LAYER
        #         filter_shape = [filter_size, 128, 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        #         conv = tf.nn.conv2d(self.lstm_out_expanded, W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
        #         # NON-LINEARITY
        #         h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="relu")
        #         # MAXPOOLING
        #         pooled = tf.nn.avg_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
        #         pooled_outputs.append(pooled)
        #
        # # COMBINING POOLED FEATURES
        # num_filters_total = num_filters * len(filter_sizes)
        # self.h_pool = tf.concat(pooled_outputs, 3)
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # #3. DROPOUT LAYER ###################################################################
        # with tf.name_scope("dropout"):
        #     self.h_drop = tf.nn.dropout(last, self.dropout_keep_prob)
        #
        # # Final (unnormalized) scores and predictions
        # with tf.name_scope("output"):
        #     W = tf.get_variable(
        #         "W",
        #         shape=[num_filters_total, num_classes],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        #     l2_loss += tf.nn.l2_loss(W)
        #     l2_loss += tf.nn.l2_loss(b)
        #     self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        #     # self.predictions = tf.argmax(self.scores, 1, name="predictions")
        #     self.predictions = tf.cast((tf.nn.sigmoid(self.scores) > 0.5), tf.float32)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


        print("(!!) LOADED LSTM-CNN! :)")
        #embed()



# 1. Embed --> LSTM
# 2. LSTM --> CNN
# 3. CNN --> Pooling/Output