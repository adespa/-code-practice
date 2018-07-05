import os
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

os.chdir('/Users/finogeeks/Documents/learn_datas/weibo_seg')


out_xfile = 'train_data/train_x.txt'

max_length = 80

PAD = 'PAD'
UNK = 'UNK'
start_voc = [PAD, UNK]

def get_vocabulary(in_file, out_file):
    voca = {}
    max_line = 0
    with open(in_file, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip()
            for word in line:
                if word in voca:
                    voca[word] += 1
                else:
                    voca[word] = 1
        voca_list = start_voc + sorted(voca, key=voca.get, reverse=True)
        if len(voca_list) > 5000:
            voca_list = voca_list[:5000]
        print(in_file + 'vocabulary size:', len(voca_list))
        with open(out_file, 'w', encoding='utf-8') as f2:
            for word in voca_list:
                f2.write(word + '\n')
    print(max_line)


voca_file = 'train_data/vocabulary.txt'
# get_vocabulary(out_xfile, voca_file)


def conver_to_vect(in_file, voca_file, out_file):
    tmp_vocab = []
    with open(voca_file, 'r', encoding='utf-8') as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    print('test this word:', vocab.get('notis', 1))
    output_f = open(out_file, 'w', encoding='utf-8')
    with open(in_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            line_vec = []
            for word in line.strip():
                line_vec.append(vocab.get(word, 1))
            output_f.write(' '.join([str(num) for num in line_vec]) + '\n')
        output_f.close()


vect_xfile = 'train_data/x_vect.txt'
yfile = 'train_data/train_y.txt'

# conver_to_vect(out_xfile, voca_file, vect_xfile)

def get_data():
    data_x = []
    data_y = []
    with open(vect_xfile, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip()
            new_line = [int(num) for num in line.split()]
            data_x.append(new_line)
    with open(yfile, 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            line = line.strip()
            data_y.append(int(line))
    return data_x, data_y
    # idxs = [i for i in range(len(data_y))]
    # random.shuffle(idxs)
    # new_datax = []
    # new_datay = []
    # for idx in idxs:
    #     new_datax.append(data_x[idx])
    #     new_datay.append(data_y[idx])
    # return new_datax, new_datay

def main():
    data_x, data_y = get_data()
    print(len(data_x), len(data_y))
    data_x = sequence.pad_sequences(data_x, maxlen=max_length)
    # print(data_x[0])
    xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=0.2)

    EMBEDDING_SIZE = 256
    HIDDEN_LAYER_SIZE = 128

    model = Sequential()
    model.add(Embedding(5000, EMBEDDING_SIZE, input_length=max_length))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    if os.path.exists(os.getcwd() + '/model/weights.best.hdf5'):
        model.load_weights("weights.best.hdf5")
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    filepath="model/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    model.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=callbacks_list, validation_data=(xtest, ytest))

main()
