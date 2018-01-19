#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

'''
torch.manual_seed(1)
lstm = nn.LSTM(3, 3) 
inputs = [Variable(torch.randn((1, 3))) for _ in range(5)]
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (Variable(torch.randn(1, 1, 3)), Variable(torch.randn(1, 1, 3))) # clean out hidden state
out, hidden = lstm(inputs, hidden)
print('out:', out)
print('hidden:', hidden)
'''

EMBEDDING_DIM = 3
HIDDEN_DIM = 5

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    
word_to_ix = {} # 单词的索引字典
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
#print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2} # 手工设定词性标签数据字典

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
 
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        #nn.Embedding(vocab_size,embed_dim)类是Module类的子类，这里它接受最重要的两个初始化参数：词汇量大小，每个词汇向量表示的向量维度。Embedding类返回的是一个以索引表示的大表，表内每一个索引对应的元素都是表示该索引指向的单词的向量表示，大表具体是以矩阵的形式存储的。这就是著名的词向量。
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
 
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
 
    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))
 
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
        

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)        

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(training_data[0][0])
print(inputs)
print(tag_scores)

for epoch in range(300): # 我们要训练300次，可以根据任务量的大小酌情修改次数。
    for sentence, tags in training_data:
    # 清除网络先前的梯度值，梯度值是Pytorch的变量才有的数据，Pytorch张量没有
        model.zero_grad()
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰
        model.hidden = model.init_hidden()
        # 准备网络可以接受的的输入数据和真实标签数据，这是一个监督式学习
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # 运行我们的模型，直接将模型名作为方法名看待即可
        tag_scores = model(sentence_in)
        # 计算损失，反向传递梯度及更新模型参数
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
     
# 来检验下模型训练的结果
inputs = prepare_sequence('Everybody ate the apple'.split(), word_to_ix)
tag_scores = model(inputs)
print(tag_scores)
