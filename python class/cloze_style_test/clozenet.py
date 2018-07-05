import numpy as np
import tensorflow as tf
from collections import defaultdict
from .netclass import BiGRURnn, InferRnn
from .load_data import get_test_batch, get_next_batch


class ClozeNet(object):
    """ClozeNet.
    Args:
        batch_size: The batch size of the train data.
        content_length: for each train sample, the length of the content,
                        the value is the maximum length of the content in the train data.
        question_length: the maximum length of the question in the train data.
        vocab_size: the number of the word in the train data without repeat.

    """

    def __init__(self, batch_size, content_length, question_length, vocab_size):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.int32, [batch_size, content_length])
        self.Q = tf.placeholder(tf.int32, [batch_size, question_length])
        self.A = tf.placeholder(tf.int32, [batch_size])
        self.predict = None

    def build_net(self, embedding_dim, encoding_dim):
        embeddings = tf.Variable(tf.random_normal([self.vocab_size, embedding_dim], stddev=0.22), dtype=tf.float32)
        tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), [embeddings])

        with tf.variable_scope('encode'):
            X_lens = tf.reduce_sum(tf.sign(tf.abs(self.X)), 1)
            embedded_X = tf.nn.embedding_lookup(embeddings, self.X)
            BidirectionrnnX = BiGRURnn(self.keep_prob, encoding_dim)
            encoded_X = BidirectionrnnX.birnn(embedded_X, X_lens, 'X')

            Q_lens = tf.reduce_sum(tf.sign(tf.abs(self.Q)), 1)
            embedded_Q = tf.nn.embedding_lookup(embeddings, self.Q)
            BidirectionrnnQ = BiGRURnn(self.keep_prob, encoding_dim)
            encoded_Q = BidirectionrnnQ.birnn(embedded_Q, Q_lens, 'Q')

        g_q = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)
        g_d = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)

        infer_rnn = InferRnn(self.keep_prob, 4 * encoding_dim)
        d_attention = infer_rnn.inferrnn(8, encoded_Q, encoded_X, g_d, g_q)
        # print(d_attention[1])
        self.predict = tf.to_float(tf.sign(tf.abs(self.X))) * d_attention

    def train_net(self, step_num):
        X_attentions = self.predict
        loss = -tf.reduce_mean(tf.log(tf.reduce_sum(tf.to_float(tf.equal(tf.expand_dims(self.A, -1), self.X))
                                                    * X_attentions, 1) + tf.constant(0.00001)))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads_and_vars = optimizer.compute_gradients(loss)
        capped_grads_and_vars = [(tf.clip_by_norm(g, 5), v) for g, v in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_grads_and_vars)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Restore the last train option
            ckpt = tf.train.get_checkpoint_state('.')
            if ckpt != None:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("no model found!")

            for step in range(step_num):
                train_x, train_q, train_a = get_next_batch()
                loss_, _ = sess.run([loss, train_op],
                                    feed_dict={self.X: train_x, self.Q: train_q, self.A: train_a, self.keep_prob: 0.7})
                print(loss_)

                # Save the model and calculate the accuracy
                if step % 1000 == 0:
                    path = saver.save(sess, './machine_reading.model', global_step=step)
                    print(path)

                    test_x, test_q, test_a = get_test_batch()
                    test_x, test_q = np.array(test_x[:self.batch_size]), np.array(test_q[:self.batch_size])
                    test_a = np.array(test_a[:self.batch_size])
                    attentions = sess.run(X_attentions, feed_dict={self.X: test_x, self.Q: test_q, self.keep_prob: 1.})
                    correct_count = 0
                    for x in range(test_x.shape[0]):
                        probs = defaultdict(int)
                        for idx, word in enumerate(test_x[x, :]):
                            probs[word] += attentions[x, idx]
                        guess = max(probs, key=probs.get)
                        if guess == test_a[x]:
                            correct_count += 1
                    print(correct_count / test_x.shape[0])
