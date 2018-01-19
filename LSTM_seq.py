#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


use_gpu = torch.cuda.is_available()

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

word_to_idx = {} # 单词的索引字典
tag_to_idx = {} # 词性的索引字典
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for label in tags:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)
            
#tag_to_idx = {"DET": 0, "NN": 1, "V": 2} # 手工设定词性标签数据字典

# 对字母按顺序进行编码
alphabet = 'abcdefghijklmnopqrstuvwxyz'
character_to_idx = {}
for i in range(len(alphabet)):
    character_to_idx[alphabet[i]] = i

    
def prepare_sequence(seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

    
class CharLSTM(nn.Module):
    #传入n个字符，通过nn.Embedding得到词向量，接着传入LSTM网络，得到状态输出h
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True)

    def forward(self, x):
        x = self.char_embedding(x)
        _, h = self.char_lstm(x)
        return h[0]
    
class LSTMTagger(nn.Module):
 
    def __init__(self, vocab_size, embedding_dim, n_char, char_dim, char_hidden,
                n_hidden, n_tag):
        super(LSTMTagger, self).__init__()
 
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        """nn.Embedding类是Module类的子类,它接受两个初始化参数:
        词汇量大小：vocab_size,
        每个词汇向量表示的向量维度：embedding_dim。
        Embedding类返回的是一个以索引表示的大表,
        表内每一个索引对应的元素都是表示该索引指向的单词的向量表示，
        大表具体是以矩阵的形式存储的。这就是词向量。"""
        
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        #n_char和char_dim表示字符的词向量维度,char_hidden表示CharLSTM输出的维度
        self.lstm = nn.LSTM(embedding_dim + char_hidden,
                            n_hidden, batch_first=True)
        #n_hidden表示每个单词作为序列输入的LSTM输出维度
        self.linear = nn.Linear(n_hidden, n_tag)
        #n_tag表示输出的词性种类     
                
    def forward(self, x, word):
        #word = [i for i in word_data]
        #将所有单词传入CharLSTM，得到字符的词向量char
        char = torch.FloatTensor()
        for each in word:
            word_list = []
            for letter in each:
                word_list.append(character_to_idx[letter.lower()])
            word_list = torch.LongTensor(word_list)
            word_list = word_list.unsqueeze(0) #LSTM的输出要带上batch_size维度
            #if use_gpu:
            #    tempchar = self.char_lstm(Variable(word_list).cuda())
            #else:
            tempchar = self.char_lstm(Variable(word_list))    
            tempchar = tempchar.squeeze(0) 
            char = torch.cat((char, tempchar.cpu().data), 0)
        char = char.squeeze(1)
        char = Variable(char)#.cuda()
        
        x = self.word_embeddings(x) #单词的词向量
        
        x = torch.cat((x, char), 1)       
        # 将char和单词的词向量拼在一起形成一个新的输入
        x = x.unsqueeze(0)    
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.linear(x)
        y = F.log_softmax(x)
        return y

        
model = LSTMTagger(
    len(word_to_idx), len(character_to_idx), 26, 100, 50, 128 ,len(tag_to_idx))
    
#if use_gpu:
#    model.cuda()
    
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)        
'''
inputs = prepare_sequence(training_data[0][0], word_to_idx)
#if use_gpu:
#    inputs = inputs.cuda()
tag_scores = model(inputs,"The dog ate the apple".split())
print(training_data[0][0])
print(inputs)
print(tag_scores)
'''
for epoch in range(300): # 我们要训练300次，可以根据任务量的大小酌情修改次数。

    running_loss = 0
    for data in training_data:
        word, tags = data
        # 准备网络可以接受的的输入数据和真实标签数据，这是一个监督式学习
        word_list = prepare_sequence(word, word_to_idx)
        tags = prepare_sequence(tags, tag_to_idx)
        #if use_gpu:
        #    sentence_in = sentence_in.cuda()
        #    targets = targets.cuda()
            
        # forward
        tag_scores = model(word_list, word)
        loss = loss_function(tag_scores, tags)
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('epoch:', epoch + 1 ,'loss:', running_loss /len(data)) 
    
# 检验模型训练的结果
inputs = prepare_sequence('Everybody ate the apple'.split(), word_to_idx)
#if use_gpu:
#    inputs = inputs.cuda()
tag_scores = model(inputs, 'Everybody ate the apple'.split())
print(tag_scores)
