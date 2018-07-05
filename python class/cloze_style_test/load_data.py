import os
import pickle
import ast

os.chdir('/Users/finogeeks/Documents/learn_datas')

train_data = 'train.vec'
valid_data = 'valid.vec'

word2idx, content_length, question_length, vocab_size = pickle.load(open('vocab.data', "rb"))
print(content_length, question_length, vocab_size)

batch_size = 16

train_file = open(train_data)


def get_next_batch():
    X = []
    Q = []
    A = []
    for i in range(batch_size):
        for line in train_file:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
            break

    if len(X) == batch_size:
        return X, Q, A
    else:
        train_file.seek(0)
        return get_next_batch()


def get_test_batch():
    with open(valid_data) as f:
        X = []
        Q = []
        A = []
        for line in f:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
        return X, Q, A