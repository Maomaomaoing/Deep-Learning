# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:42:45 2019

@author: MaoChuLin
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from collections import Counter
import json
import pandas as pd
from nltk import word_tokenize

#%%
## pre-processing

def preprocess(accept_file, reject_file, sen_length):
    lower = lambda word: word.lower()
    
    def sen2word(data, sen_length):   
        split = lambda sen: word_tokenize(sen)
        
        data = list(map(lower, data))
        data = list(map(split, data))
        data = [row for row in data if len(row) < sen_length and len(row)>1]
        
        return data

    def embedding_padding(data, vocab, pad_size):
        encode = lambda word: vocab[word]
        data = [list(map(encode, row))+[0]*(pad_size-len(row)) for row in data]
        return data
    
    ## pre-process: lowercase, sentence to word, delete too long or short sentence
    accept, reject = sen2word(accept_file, sen_length), sen2word(reject_file, sen_length)
    total = len(accept)+ len(reject)
    print("Total:", total, "\nAccepted data:", len(accept), "\nRejected data:", len(reject))
    
    ## get vocabulary
    vocab = word_tokenize(' '.join([' '.join(row) for row in accept+reject]))
    vocab = dict(zip(set(map(lower, vocab)), range(1, len(vocab)+1)))
    print("Vocab size:", len(vocab))
    
    ## pre-process: word to embedding, padding to length = 16
    accept, reject = embedding_padding(accept, vocab, sen_length), embedding_padding(reject, vocab, sen_length)
    
    ## split train/test, test is first 50 data each class
    train_1, test_1, train_0, test_0 = accept[50:], accept[:50], reject[50:], reject[:50]
    
    return train_1, test_1, train_0, test_0, vocab



def get_batch(accept, reject, batch_size):
    
    half_size = int(batch_size/2)
    random.shuffle(accept)
    random.shuffle(reject)
    
    ## get all batch, accept is less
#    n_batch = int(len(accept)/half_size)+1
    n_batch = int(len(accept)/half_size)
    batch = [[] for _ in range(n_batch)]
    labels = [[] for _ in range(n_batch)]
    for i in range(n_batch):
        batch[i].extend( accept[i*half_size: (i+1)*half_size] + 
                         reject[i*half_size: (i+1)*half_size] )
        labels[i].extend([1]*half_size + [0]*half_size)
        
    ### less than a batch, random choose
#    batch[n_batch-1] = random.choices(accept, k=half_size)+random.choices(reject, k=half_size)
#    labels[n_batch-1] = [1]*half_size + [0]*half_size
    
    ## concate accept and reject
#    batch = [accept + reject]
#    labels = [[1]*len(accept) + [0]*len(reject)]
    
    ## shuffle the data in a batch(origin: 1 first, 0 last, shuffle to random)    
    for i_batch, label in zip(batch, labels):
        seed = random.randint(1,300)
        random.seed(seed)
        random.shuffle(i_batch)
        random.seed(seed)
        random.shuffle(label)
  
    return batch, labels

def save_json(data, filename):
    to_float = lambda num: float(num)
    with open(filename, 'w') as f:
        json.dump(list(map(to_float, data)), f)
    print("Save file at", filename)

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
   
""" model """

class RNN_Model():
   
    def __init__(self, optm, lr, batch_size, k, vocab_size, sen_length):
        
        
        self.lr = lr
        self.k = k
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.sen_length = sen_length
        self.optimizer = optm
        self.initial = tf.keras.initializers.he_normal()
        
        
        self.sens = tf.placeholder(tf.int64, [None, sen_length]) ## batch, sen_len
        self.y = tf.placeholder(tf.int64, [None])
        print("Sentence shape", self.sens.shape)
        
    
        word_embeddings = tf.get_variable("word_embeddings", [vocab_size, 50])
        self.embedded = tf.nn.embedding_lookup(word_embeddings, self.sens)
        print("Embedding shape", self.embedded.shape)
        
#        self.sens = tf.reshape(self.sens, [-1, sen_length, 1])
    
    
        self.output = self.forward(3, 32, self.embedded)
        self.prediction = tf.argmax(self.output, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))
        
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.y, self.k), self.output)        
        gradients = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(gradients)
        
## 1694         
    def forward(self, num_layers, num_cells, input_):
        
        def get_a_cell(lstm_size):
            lstm = tf.contrib.rnn.BasicRNNCell(lstm_size, activation = 'tanh')
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.7)
            return lstm
        self.cell = tf.contrib.rnn.MultiRNNCell(
                [get_a_cell(num_cells) for _ in range(num_layers)]
            )
        
#        self.rnn_cell = tf.contrib.rnn.BasicRNNCell(10, activation='tanh')
        _outputs, _ = tf.nn.dynamic_rnn(cell=self.cell, inputs=input_, dtype=tf.float32)
        print("Rnn output shape", _outputs.shape)
        self.output = _outputs[:, -1]
#        self.output = tf.layers.dense(self.output, 16, activation='relu')
        self.output = tf.layers.dense(self.output, self.k)
        
        return self.output


#print("GPU available:", tf.test.is_gpu_available(), tf.test.gpu_device_name())




#%%

#widths = []
#heights = []
#  
#""" preprocessing """
### get all image's shape
#for category in categories: 
#    for i, file in enumerate(train_file[category]):
#        img = plt.imread(file, format=None)
#
#        widths.append(img.shape[1])
#        heights.append(img.shape[0])
#        print(category, i)
#
#widths = np.array(widths)
#heights = np.array(heights)
#
#ratio = widths / heights
#
### decide height, width, choose (h, w) = (225, 300)
#np.percentile(widths, 10), np.percentile(widths, 50), np.percentile(widths, 90)
#plt.boxplot(widths[widths<1000])
#
#np.percentile(heights, 10), np.percentile(heights, 50), np.percentile(heights, 90)        
#plt.boxplot(heights[heights<1000])
#temp = Counter(heights)
#
#np.percentile(ratio, 25), np.percentile(ratio, 50), np.percentile(ratio, 75)        
#plt.boxplot(ratio)


#%%


""" load data """
k = 2 ## k = 2 categories

batch_size = 100
sen_length = 10

#n1_batch = int(train_total / batch_size)
#n2_batch = int(test_total / batch_size)

## load data
#os.chdir("D:/School/碩一/深度學習/DL_HW2")
#os.chdir("/home/0756728/DL_hw2/")
accept_file = list(pd.read_excel("ICLR_accepted.xlsx")[0])
reject_file = list(pd.read_excel("ICLR_rejected.xlsx")[0])

## accept: 1, reject: 0
train_1, test_1, train_0, test_0, vocab = preprocess(accept_file, reject_file, sen_length)

vocab_size = len(vocab)

## select sentence length
#sentence_len = sorted([len(row) for row in accept+reject], reverse = True)
#plt.boxplot(sentence_len)

## sentence max length = 10, embedding size = 1694
## a sentence = 10*2394 = 16940



#%%
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ['CUDA_VISIBLE_DEVICES']= '0'


""" run """
lr = 1e-4
n_epoch = 200
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
tf.reset_default_graph()
model = RNN_Model(optimizer, lr, batch_size, k, vocab_size, sen_length)

train_acc, train_loss, test_acc, test_loss = [], [], [], []
saver = tf.train.Saver()


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    for epoch in range(n_epoch):
        
        ## get all batch
        train_batch, train_labels = get_batch(train_1, train_0, batch_size)
        n1_batch = len(train_batch)
        test_batch, test_labels = get_batch(test_1, test_0, batch_size)
        n2_batch = len(test_batch)
        
        
        """train"""
        for i, (batch, labels) in enumerate(zip(train_batch, train_labels)):
            batch = np.array(batch).reshape([-1, sen_length])
            acc, _, loss = sess.run([model.accuracy, model.train_op, model.loss], 
                                    {model.sens: batch, model.y:labels})
        
        if epoch % 10 ==0:
            print("Epoch %s: train acc %s, loss %s" % (epoch, acc, loss))
        train_acc.append(acc); train_loss.append(loss)
        
        """ test """
        batch_acc = 0
        for i, (batch, labels) in enumerate(zip(test_batch, test_labels)):
            batch = np.array(batch).reshape([-1, sen_length])
            acc, predict, loss = sess.run([model.accuracy, model.prediction, model.loss], 
                                    {model.sens: batch, model.y:labels})
#            batch_acc += acc
        if epoch % 10 ==0:
            print("Epoch %s: test acc %s"% (epoch, acc))
        test_acc.append(acc); test_loss.append(loss)
    
    save_path = saver.save(sess, "model.ckpt")
    
""" save file """
save_json(train_acc, "train_acc.json")
save_json(train_loss, "train_loss.json")

save_json(test_acc, "test_acc.json")
save_json(test_loss, "test_loss.json")

to_float = lambda num: float(num)
plt.plot(range(len(train_loss)), list(map(to_float, train_loss)))
#plt.plot(range(len(test_loss)), list(map(to_float, test_loss)))
plt.show()
plt.plot(range(len(train_acc)), list(map(to_float, train_acc)))
plt.plot(range(len(test_acc)), list(map(to_float, test_acc)))
plt.show()
#%%
""" test """


#with tf.Session() as sess:
#    init_op = tf.global_variables_initializer()
#    sess.run(init_op)
#    embedded = sess.run([model.embedded], {model.sens: batch, model.y:labels})

##
#filename = "D:/School/碩一/深度學習/DL_HW2/animal-10/train/butterfly/e83cb80e29f11c22d2524518b7444f92e37fe5d404b0144390f8c770a3e5b7_640.jpg"
##img2 = plt.imread(filename)
#img = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
##img = tf.reshape(
##        tf.image.resize_image_with_crop_or_pad(
##                tf.image.resize_images(img, [300, 300], preserve_aspect_ratio = True)
##                , 300, 300)
##        , [300, 300, 3])
#
#if img.shape[1]<301 and img.shape[0]<301:
#    img2 = tf.reshape(
#        tf.image.resize_image_with_crop_or_pad(img, 300, 300)
#        , [300, 300, 3])
#elif img.shape[1]>img.shape[0]: ## 200, 400 (r=2)-> 150, 300
#    ratio = img.shape[1] / img.shape[0]
#    img2 = tf.image.resize_images(img, [int(300/ratio), 300])
#    print(img2.shape)
#elif img.shape[0]>img.shape[1]: ## 400, 200 (r=0.5)-> 300, 150
#    ratio = img.shape[1] / img.shape[0]
#    img2 = tf.image.resize_images(img, [300, int(300*ratio)])
#
#
#with tf.Session() as sess:
##    img1 = sess.run([model.imgs], {model.x_path: train_file['cat']})
#    img1 = sess.run(img2)
##    
##plt.imshow(plt.imread(filename))
##plt.imshow(img1.astype('uint8'))


## embedding lookup