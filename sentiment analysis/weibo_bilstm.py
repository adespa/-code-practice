import yaml
import os
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd
import sys


os.chdir('/Users/finogeeks/Documents/learn_datas/weibo_seg')
vocab_dim = 200
maxlen = 80
n_iterations = 5  
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 80
cpu_count = multiprocessing.cpu_count()

xfile = 'train_data/train_x.txt'
yfile = 'train_data/train_y.txt'


def loadfile():
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


def tokenizer(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


def create_dictionaries(model=None, combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec.load('lstm_data/Word2vec_model2.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(len(x_train),len(y_train))
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_bilstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(Bidirectional(LSTM(units=128)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # print('loading weight...')
    # model.load_weights('lstm_data/lstm.h5')
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('lstm_data/lstm.h5')
    print('Test score:', score)


#训练模型，并保存
def train():
    print('Loading Data...')
    combined,y=loadfile()
    print(len(combined),len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print(len(x_train),len(y_train))
    train_bilstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)
