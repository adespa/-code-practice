# import csv
# import re
# import random
import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

from IPython import embed


def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

def create_dictionaries(model=None, combined=None, maxlen=80):
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
def word2vec_train(combined, maxlen):

    # model = Word2Vec(size=vocab_dim,
    #                  min_count=n_exposures,
    #                  window=window_size,
    #                  workers=cpu_count,
    #                  iter=n_iterations)
    # model.build_vocab(combined)
    # model.train(combined, total_examples=model.corpus_count, epochs=5)
    # model.save('lstm_data/Word2vec_model2.pkl')
    model = Word2Vec.load('lstm_data/Word2vec_model2.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined, maxlen=maxlen)
    return index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y,vocab_dim):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.01)
    print(len(x_train),len(y_train))
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


def get_test_dataset(combined, w2indx):
    ''' Words become integers
    '''
    data = []
    for sentence in combined:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    data = sequence.pad_sequences(data, maxlen=80)
    return data

# #Separates a file with mixed positive and negative examples into two.
# def separate_dataset(filename):
#     good_out = open("good_"+filename,"w+");
#     bad_out  = open("bad_"+filename,"w+");
#
#     seen = 1;
#     with open(filename,'r') as f:
#         reader = csv.reader(f)
#         reader.next()
#
#         for line in reader:
#             seen +=1
#             sentiment = line[1]
#             sentence = line[3]
#
#             if (sentiment == "0"):
#                 bad_out.write(sentence+"\n")
#             else:
#                 good_out.write(sentence+"\n")
#
#             if (seen%10000==0):
#                 print(seen);
#
#     good_out.close();
#     bad_out.close();
#
#
#
# #Load Dataset
# def get_dataset(goodfile,badfile,limit,randomize=True):
#     good_x = list(open(goodfile,"r").readlines())
#     good_x = [s.strip() for s in good_x]
#
#     bad_x  = list(open(badfile,"r").readlines())
#     bad_x  = [s.strip() for s in bad_x]
#
#     if (randomize):
#         random.shuffle(bad_x)
#         random.shuffle(good_x)
#
#     good_x = good_x[:limit]
#     bad_x = bad_x[:limit]
#
#     x = good_x + bad_x
#     x = [clean_str(s) for s in x]
#
#
#     positive_labels = [[0, 1] for _ in good_x]
#     negative_labels = [[1, 0] for _ in bad_x]
#     y = np.concatenate([positive_labels, negative_labels], 0)
#     return [x,y]
#
#
#
#
# #Clean Dataset
# def clean_str(string):
#
#
#     #EMOJIS
#     string = re.sub(r":\)","emojihappy1",string)
#     string = re.sub(r":P","emojihappy2",string)
#     string = re.sub(r":p","emojihappy3",string)
#     string = re.sub(r":>","emojihappy4",string)
#     string = re.sub(r":3","emojihappy5",string)
#     string = re.sub(r":D","emojihappy6",string)
#     string = re.sub(r" XD ","emojihappy7",string)
#     string = re.sub(r" <3 ","emojihappy8",string)
#
#     string = re.sub(r":\(","emojisad9",string)
#     string = re.sub(r":<","emojisad10",string)
#     string = re.sub(r":<","emojisad11",string)
#     string = re.sub(r">:\(","emojisad12",string)
#
#     #MENTIONS "(@)\w+"
#     string = re.sub(r"(@)\w+","mentiontoken",string)
#
#     #WEBSITES
#     string = re.sub(r"http(s)*:(\S)*","linktoken",string)
#
#     #STRANGE UNICODE \x...
#     string = re.sub(r"\\x(\S)*","",string)
#
#     #General Cleanup and Symbols
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#
#     return string.strip().lower()
#
#
#
# #Generate random batches
# #Source: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
# def gen_batch(data, batch_size, num_epochs, shuffle=True):
#     """
#     Generates a batch iterator for a dataset.
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
#     for epoch in range(num_epochs):
#         # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]
#
# if __name__ == "__main__":
#     separate_dataset("small.txt");
#
#
# #42
# #642