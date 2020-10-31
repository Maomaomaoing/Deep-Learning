# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:45:01 2019

@author: MaoChuLin
"""
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

#embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
#embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
#print(tf.shape(x), x.shape)
#initial = tf.keras.initializers.he_normal()
#tf.zeros_initializer()

class Model():
    def __init__(self, k):
        
        self.x = tf.placeholder(tf.float32, [None, 4, 1])
        self.y = tf.placeholder(tf.float32, [None, k])
        self.rnn_cell = tf.contrib.rnn.BasicRNNCell(2, activation='tanh')
        _outputs, _ = tf.nn.dynamic_rnn(cell=self.rnn_cell, inputs=self.x, dtype=tf.float32)
        self.output = _outputs[:, -1]
        self.output = tf.layers.dense(self.output, 2, activation='relu')
        self.prediction = tf.argmax(self.output, 1)
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.output)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        gradients = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(gradients)
        
input1 = np.array([[1,2,3,4], [4,3,2,1], [0,1,2,3], [3,2,1,0], [2,3,4,5], [5,4,3,2]]).astype(np.float32).reshape([6,4,1])
label = np.array([[0, 1], [1,0], [0,1], [1,0], [0,1], [1,0]]).astype(np.uint8)
k = 2

epoch = 10000
model = Model(k)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    _, loss, output = sess.run([model.train_op, model.loss, model.output], {model.x:input1, model.y: label})
    prediction = sess.run([model.prediction], {model.x: np.array([6,5,4,3]).reshape(1,4,1)})
            

            





