import tensorflow as tf


class Glimpse(object):

    def __init__(self, keep_prob, stddev):
        self.keep_prob = keep_prob
        self.stddev = stddev

    def glimpse(self, encodings, inputs, name):
        with tf.variable_scope(name):
            w_c = inputs.get_shape()[1]
            w_r = encodings.get_shape()[2]
            weights = tf.Variable(tf.random_normal([int(w_r), int(w_c)], stddev=self.stddev), dtype=tf.float32)
            bias = tf.Variable(tf.random_normal([int(w_r), 1], stddev=self.stddev), dtype=tf.float32)
            weights = tf.nn.dropout(weights, self.keep_prob)
            inputs = tf.nn.dropout(inputs, self.keep_prob)
            attention = tf.transpose(tf.matmul(weights, tf.transpose(inputs)) + bias)
            attention = tf.matmul(encodings, tf.expand_dims(attention, -1))
            attention = tf.nn.softmax(tf.squeeze(attention, -1))
        return attention, tf.reduce_sum(tf.expand_dims(attention, -1) * encodings, 1)


class BiGRURnn(object):

    def __init__(self, keep_prob, encoding_dim):
        self.keep_prob = keep_prob
        self.encoding_dim = encoding_dim

    def birnn(self, vec_inp, lens, name):
        with tf.variable_scope(name):
            dro_inp = tf.nn.dropout(vec_inp, self.keep_prob)
            gru_cell = tf.nn.rnn_cell.GRUCell(self.encoding_dim)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, dro_inp,
                                                                     sequence_length=lens, dtype=tf.float32,
                                                                     swap_memory=True)
            encoded = tf.concat(outputs, 2)
        return encoded


class InferRnn(object):

    def __init__(self, keep_prob, cell_dim):
        self.keep_prob = keep_prob
        self.cell_dim = cell_dim

    def inferrnn(self, rnn_len, encoded_Q, encoded_X, g_d, g_q):
        with tf.variable_scope('attend') as scope:
            infer_gru = tf.nn.rnn_cell.GRUCell(self.cell_dim)
            infer_state = infer_gru.zero_state(encoded_Q.get_shape()[0], tf.float32)
            for iter_step in range(rnn_len):
                if iter_step > 0:
                    scope.reuse_variables()

                Doglimpse = Glimpse(self.keep_prob, 0.22)
                _, q_glimpse = Doglimpse.glimpse(encoded_Q, infer_state, 'glimpse_q')
                d_attention, d_glimpse = Doglimpse.glimpse(encoded_X, tf.concat([infer_state, q_glimpse], 1), 'glimpse_d')

                gate_concat = tf.concat([infer_state, q_glimpse, d_glimpse, q_glimpse * d_glimpse], 1)

                # w_r = gate_concat.get_shape()[1]
                # w_c = encoded_Q.get_shape()[2]
                # print(w_r, w_c)

                # g_q = tf.Variable(tf.random_normal([int(w_r), int(w_c)], stddev=0.22),
                #                   dtype=tf.float32)
                # g_d = tf.Variable(tf.random_normal([int(w_r), int(w_c)], stddev=0.22),
                #                   dtype=tf.float32)

                r_d = tf.sigmoid(tf.matmul(gate_concat, g_d))
                r_d = tf.nn.dropout(r_d, self.keep_prob)
                r_q = tf.sigmoid(tf.matmul(gate_concat, g_q))
                r_q = tf.nn.dropout(r_q, self.keep_prob)

                combined_gated_glimpse = tf.concat([r_q * q_glimpse, r_d * d_glimpse], 1)
                _, infer_state = infer_gru(combined_gated_glimpse, infer_state)
        return d_attention