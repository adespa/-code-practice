import jieba
import os
import re
import numpy as np
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import scale
from  sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

os.chdir('/Users/finogeeks/Documents/learn_datas/weibo_seg')


weibo_file = 'weibo_emo_x.txt'

def clean_data(in_file, out_file):
    f1 = open(in_file, 'r', encoding='utf-8')
    f2 = open(out_file, 'w', encoding='utf-8')
    for line in f1.readlines():
        line = line.strip()
        patter = "[\s+\.\/_,$%^*(+\"\']+|[+——！，。、~@#￥%……&*（）：＞＜“”；>一【】＂\)]"
        new_line = re.sub(patter, '', line)
        sline = jieba.cut(new_line)
        # new_sline = [w for w in sline if w not in stop_word]
        f2.write(' '.join(sline))
    f1.close()
    f2.close()

out_file = 'weibo_test_x.txt'

# 清理数据
# clean_data(weibo_file, out_file)

# pos_dic = ['哈哈', '偷笑']
# neg_dic = ['泪','抓狂','汗','挖鼻屎','怒','衰','晕','可怜', '委屈']

# pos_dic = ['太开心']
# neg_dic = ['悲伤','失望','怒骂','伤心']
pos_dic = ['威武']
neg_dic = []


def get_train_data(xfile, yfile, out_xfile, out_yfile):
    f1 = open(xfile, 'r', encoding='utf-8')
    f2 = open(yfile, 'r', encoding='utf-8')
    f3 = open(out_xfile, 'w', encoding='utf-8')
    f4 = open(out_yfile, 'w', encoding='utf-8')
    pos_num = 0
    neg_num = 0
    xlines = f1.readlines()
    yline = f2.readlines()
    if len(xlines) != len(yline):
        print('not match')
        return None
    for i in range(len(yline)):
        if yline[i].strip() in pos_dic:
            if pos_num < 10000:
                line = xlines[i].strip()
                patter = "[\s+\.\/_,$%^*(+\"\']+|[+——！，。、~@#￥%……&*（）：＞＜“”；>一【】＂\)]"
                new_line = re.sub(patter, '', line)
                if len(new_line) < 80:
                    f3.write(new_line + '\n')
                    f4.write('1\n')
                    pos_num += 1
        elif yline[i].strip() in neg_dic:
            if neg_num < 10000:
                line = xlines[i].strip()
                patter = "[\s+\.\/_,$%^*(+\"\']+|[+——！，。、~@#￥%……&*（）：＞＜“”；>一【】＂\)]"
                new_line = re.sub(patter, '', line)
                if len(new_line) < 80:
                    f3.write(new_line + '\n')
                    f4.write('0\n')
                    neg_num += 1
    f1.close()
    f1.close()
    f3.close()
    f4.close()
    print(pos_num, neg_num)


yfile = 'weibo_emo_y.txt'
out_xfile = 'train_data/train_x.txt'
out_yfile = 'train_data/train_y.txt'


# 获得指定标签训练集
# get_train_data(weibo_file, yfile, out_xfile, out_yfile)
get_train_data(weibo_file, yfile, 'train_data/model_test_x1.txt', 'train_data/model_test_y1.txt')


# 训练模型
n_dim = 300
# model = word2vec.Word2Vec(sentences,size=n_dim, min_count=10, sg=1)
# # imdb_w2v = Word2Vec(size=n_dim, min_count=10, sg=1)
# # imdb_w2v.build_vocab(corpus)
# model.save('w2v_model/w2v_model.pkl')

# model = word2vec.Word2Vec.load('w2v_model/w2v_model.pkl')
# print(model.similarity('这里', ''))

x_train_f = 'train_data/x_train.txt'
y_train_f = 'train_data/y_train.txt'
x_test_f = 'train_data/x_test.txt'
y_test_f = 'train_data/y_test.txt'
def split_data():
    f1 = open(out_xfile, 'r', encoding='utf-8')
    f2 = open(out_yfile, 'r', encoding='utf-8')
    train_x = []
    train_y = []
    for line in f1.readlines():
        line = line.strip()
        patter = "[\s+\.\/_,$%^*(+\"\']+|[+——！，。、~@#￥%……&*（）：＞＜“”；>一【】＂\)]"
        new_line = re.sub(patter, '', line)
        train_x.append(new_line)
    for line in f2.readlines():
        train_y.append(int(line.strip()))
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1)
    f1 = open(x_train_f, 'w', encoding='utf-8')
    f2 = open(x_test_f, 'w', encoding='utf-8')
    f3 = open(y_train_f, 'w', encoding='utf-8')
    f4 = open(y_test_f, 'w', encoding='utf-8')
    for line in x_train:
        f1.write(line + '\n')
    for line in x_test:
        f2.write(line + '\n')
    for line in y_train:
        f3.write(str(line) + '\n')
    for line in y_test:
        f4.write(str(line) + '\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    return x_train, x_test, y_train, y_test

# 划分训练集
# x_train, x_test, y_train, y_test = split_data()
# print(len(x_train), len(x_test))


# wb_model = word2vec.Word2Vec.load('w2v_model/w2v_model.pkl')
def build_vecs(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += wb_model[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def get_vecs():
    f1 = open(x_train_f, 'r', encoding='utf-8')
    f2 = open(x_test_f, 'r', encoding='utf-8')
    x_test = []
    x_train = []
    for line in f1.readlines():
        x_train.append(line.strip())
    for line in f2.readlines():
        x_test.append(line.strip())
    f1.close()
    f2.close()

    train_vecs = np.concatenate([build_vecs(z, n_dim) for z in x_train])
    train_vecs =scale(train_vecs)
    test_vecs = np.concatenate([build_vecs(z, n_dim) for z in x_test])
    test_vecs = scale(test_vecs)
    return train_vecs, test_vecs

# train_vecs, test_vecs = get_vecs()
# print (train_vecs.shape, test_vecs.shape)
train_vec_file = 'train_data/vec_train.txt'
test_vec_file = 'train_data/vec_test.txt'

def save_data(out_file1, out_file2):
    f1 = open(out_file1, 'w', encoding='utf-8')
    f2 = open(out_file2, 'w', encoding='utf-8')
    for vec in train_vecs:
        f1.write(str(vec))
        f1.write('\n')
    for vec in test_vecs:
        f2.write(str(vec))
        f2.write('\n')
    f1.close()
    f2.close()

# save_data(train_vec_file, test_vec_file)

def get_data(in_file1, in_file2):
    f1 = open(in_file1, 'r', encoding='utf-8')
    f2 = open(in_file2, 'r', encoding='utf-8')
    out_list1 = []
    out_list2 = []
    for line in f1.readlines():
        out_list1.append(int(line.strip()))
    for line in f2.readlines():
        out_list2.append(int(line.strip()))
    f1.close()
    f2.close()
    return out_list1, out_list2

