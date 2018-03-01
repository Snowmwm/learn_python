#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

from model import PoetryModel


#超参数设置
EPOCH = 10
BATCH_SIZE = 64
LR = 1e-3
#WEIGHT_DECAY = 1e-4
SEED = 666
USE_GPU = False

PRINT_EVERY = 300 #每300个batch打印一次
SAVE_EVERY = 1 #每1个epoch保存一次模型
NE = 201 #已经训练的波数
MODEL_PATH = './checkpoints/tang_201.pth'#预训练模型路径
MODEL_PREFIX = './checkpoints/tang' #模型保存路径

#生成诗歌相关配置
TRAIN = True #是否训练
ACROSTIC = True #是否藏头诗
MAX_GEN_LEN = 60 #生成诗歌最长长度
PREFIX_WORDS = '江流天地外，山色无有中。' 
START_WORDS = '深度学习' 


#随机种子设置
use_gpu = torch.cuda.is_available() and USE_GPU
torch.manual_seed(SEED)
if use_gpu:
    torch.cuda.manual_seed(SEED)

#加载数据
data = np.load('tang.zip')
data, word2ix, ix2word = data['data'], data['word2ix'].item(), \
                         data['ix2word'].item()
data = torch.from_numpy(data) #不使用dataset，直接把所有数据加载到内存中
dataloader = Data.DataLoader(data, batch_size = BATCH_SIZE,
                             shuffle = True, num_workers = 0)

#搭建神经网络
model = PoetryModel(len(word2ix), 128, 256) #字数，词向量维度，LSTM输出维度
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss() #本质上来说这就是一个分类问题

if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))

if use_gpu:
    model.cuda()
    loss_func.cuda()

#训练
def train(**kwargs):
    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data_ in enumerate(dataloader):
            data_ = data_.long().transpose(1,0).contiguous()
            if use_gpu:
                data_ = data_.cuda()
            optimizer.zero_grad()
            #预测的目标就是输入的下一个字
            input_, target = Variable(data_[:-1,:]), Variable(data_[1:,:])
            output, _ = model(input_)
            loss = loss_func(output, target.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if (i + 1) % PRINT_EVERY == 0:
                print('Epoch:', epoch+1+NE, ' Batch:', i+1,
                ' loss:%.5f' %(running_loss/(BATCH_SIZE*i)))

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    # join()方法将results列表中的字符连接起来生成诗
                    gen_poetry =  ''.join(generate(model,word,ix2word,word2ix))
                    gen_poetries.append(gen_poetry)
                print(gen_poetries)

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(),
                        '%s_%s.pth' %(MODEL_PREFIX,(epoch+1+NE)))

        gen_poetry =  ''.join(gen_acrostic(model,START_WORDS,
                                        ix2word,word2ix,PREFIX_WORDS))
        print('\n', gen_poetry)


# 给定几个词，根据这几个词接着生成一首完整的诗歌
def generate(model,start_words,ix2word,word2ix,prefix_words=None):
    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = Variable(torch.Tensor([word2ix['<START>']]).view(1,1).long())
    if use_gpu:
        input=input.cuda()
    hidden = None

    #用以控制生成诗歌的意境和长短（五言还是七言）
    if prefix_words:
        for word in prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1,1)

    for i in range(MAX_GEN_LEN):
        output,hidden = model(input,hidden)

        if i<start_word_len:
            #将start words中的字依次作为输入，计算隐藏单元
            w = results[i]
            input = Variable(input.data.new([word2ix[w]])).view(1,1)
        else:
            #用预测的字作为新的输入，计算隐藏单元和预测新的输出
            top_index  = output.data[0].topk(1)[1][0]
            w = ix2word[top_index]
            results.append(w) #将输出的字放入results列表
            input = Variable(input.data.new([top_index])).view(1,1)
        if w=='<EOP>':
            del results[-1] #遇到结束符，删除结束符然后结束
            break
    return results


#生成藏头诗
def gen_acrostic(model,start_words,ix2word,word2ix,prefix_words=None):
    results = []
    start_word_len = len(start_words)
    input = Variable(torch.Tensor([word2ix['<START>']]).view(1,1).long())
    if use_gpu:
        input=input.cuda()
    hidden = None

    index=0 # 用来指示已经生成了多少句藏头诗
    pre_word='<START>'

    if prefix_words:
        for word in prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([word2ix[word]])).view(1,1)

    for i in range(MAX_GEN_LEN):
        output,hidden = model(input,hidden)
        top_index = output.data[0].topk(1)[1][0]
        w = ix2word[top_index]

        if (pre_word  in {u'。',u'！','<START>'} ):
            # 如果是诗的开头或遇到句号、感叹号：
            if index==start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 否则把藏头的词作为输入送入模型
                w = start_words[index]
                index+=1
                input = Variable(input.data.new([word2ix[w]])).view(1,1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = Variable(input.data.new([word2ix[w]])).view(1,1)
        results.append(w)
        pre_word = w
    return results


if __name__ == '__main__':
    if TRAIN:
        train()
    if not ACROSTIC:
        gen_poetry =  ''.join(generate(model,START_WORDS,
                                        ix2word,word2ix,PREFIX_WORDS))
    else:
        gen_poetry =  ''.join(gen_acrostic(model,START_WORDS,
                                        ix2word,word2ix,PREFIX_WORDS))
    print(gen_poetry)