# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:55:13 2019

@author: MaoChuLin
"""

import os
import json
import random
import numpy as np
import tensorflow as tf

def load_filename():
    path = os.getcwd()
    filename = os.listdir(path+'/cartoon')
    filename = [path+'/cartoon/'+file for file in filename]
    return filename

def shuffle(data):
    random.shuffle(data)
    return data

def get_batch_i(filename, batch_size, i):
    
    ## get all batch, if mod != 0
    ## no label
    n = len(filename)
    if (i+1)*batch_size < n-1:
        batch = filename[i*batch_size:(i+1)*batch_size]
    else:
        batch = filename[i*batch_size:]
        batch = batch + filename[:batch_size-len(batch)]
    return batch

def save_json(data, filename):
    to_float = lambda num: float(num)
    with open(filename, 'w') as f:
        json.dump(list(map(to_float, data)), f)
    print("Save file at", filename)

class VAE_model():

    
    
    def __init__(self, batch_size, lr, optm, init):
        self.lr = lr
        self.batch_size = batch_size
        self.optm = optm
        self.init = init
        
        def img_preprocess(filename):
            img = tf.image.decode_png(tf.read_file(filename), channels=3)
    #        img = tf.image.resize_images(img, [256, 256])
            img = tf.image.resize_image_with_crop_or_pad(img, 300, 300)
            img = tf.image.resize_images(img, [64, 64])
            img = tf.reshape(img, [64, 64, 3])
            return img
        
        self.x_path = tf.placeholder(tf.string, [self.batch_size]) ## batch
        self.imgs = tf.map_fn(img_preprocess, self.x_path, dtype = tf.float32)
        
        ## normalize
        #mean, var = tf.nn.moments(self.imgs, axes = [0, 1, 2])
        #self.imgs = (self.imgs-mean) / tf.sqrt(var)
        self.imgs = (self.imgs-127.5) / 127.5
        
        self.output, self.mean, self.var, self.latent = self.forward(self.imgs)
        self.loss = self.loss_function()
        
        gradients = self.optm.compute_gradients(self.loss)
        self.train_op = self.optm.apply_gradients(gradients)
    
    def encode(self, img):
        img = tf.layers.conv2d(img, 32, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)
        img = tf.nn.relu(img) 
        img = tf.layers.conv2d(img, 64, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)
        img = tf.nn.relu(img)        
        img = tf.layers.conv2d(img, 128, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)
        img = tf.nn.relu(img)
        img = tf.layers.conv2d(img, 256, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)
        img = tf.nn.relu(img)
        img_shape = img.shape # (4, 4, 128)
        img = tf.layers.flatten(img)
        print(img.shape) # 4*4*256
        return img, img_shape
    
    def decode(self, img, img_shape):
        img = tf.layers.dense(img, 4096, activation = "relu", kernel_initializer = self.init)
        img = tf.reshape(img, img_shape)
        img = tf.layers.conv2d_transpose(img, 128, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)
        img = tf.nn.relu(img)
        img = tf.layers.conv2d_transpose(img, 64, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)
        img = tf.nn.relu(img)
        img = tf.layers.conv2d_transpose(img, 32, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)
        img = tf.nn.relu(img)
        img = tf.layers.conv2d_transpose(img, 3, 3, strides = (2,2), padding = 'same', kernel_initializer = self.init)       
        img = tf.nn.sigmoid(img)
        print(img.shape) # 64, 64, 3
        return img
    
    def bottleneck(self, latent):
        latent_size = 64
        mean = tf.layers.dense(latent, latent_size, activation = "relu", kernel_initializer = self.init)
        var = tf.layers.dense(latent, latent_size, activation = "relu", kernel_initializer = self.init)
        std = tf.reshape(tf.exp(tf.multiply(var, 0.5)), [self.batch_size, latent_size])
        eps = tf.random.normal(std.shape)
        latent = mean + std * eps
        return latent, mean, var
        
    def forward(self, input_img):
        img, img_shape = self.encode(input_img)
        latent, mean, var = self.bottleneck(img)
        img = self.decode(img, img_shape)
        return img, mean, var, latent
    
    def loss_function(self):
    
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(-0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
            
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.output, labels = self.imgs)
        logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
        logpz = log_normal_pdf(self.latent, 0., 0.)
        logqz_x = log_normal_pdf(self.latent, self.mean, self.var)
    
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

#%%

## load train data
filename = load_filename()

n = len(filename)
n_epoch = 30
batch_size = 100
n_batch = int(n/batch_size)

lr = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
#optimizer = tf.train.RMSPropOptimizer(learning_rate = lr)
initializer = tf.keras.initializers.he_normal()
model = VAE_model(batch_size, lr, optimizer, initializer)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    losses = []
    for epoch in range(n_epoch):
        
        filename = shuffle(filename)
        for i in range(n_batch): # i-th batch
            batch = get_batch_i(filename, batch_size, i)
            _, loss, output = sess.run([model.train_op, model.loss, model.output], {model.x_path: batch})
            losses.append(loss)
        print("Epoch %s: %s" % (epoch, loss))
    
    output.tofile('batch_outputs.csv',sep=',',format='%f')
    print("Save in", "batch_imgs.csv")
    save_json(losses, "loss.json")
