import os
import pickle
from .colzenet import ClozeNet

os.chdir('/Users/finogeeks/Documents/learn_datas')

word2idx, content_length, question_length, vocab_size = pickle.load(open('vocab.data', "rb"))
print(content_length, question_length, vocab_size)

batch_size = 16

cloze_net = ClozeNet(batch_size=batch_size, content_length=content_length,
                     question_length=question_length, vocab_size=vocab_size)
cloze_net.build_net(embedding_dim=384, encoding_dim=128)
cloze_net.train_net(step_num=20000)

