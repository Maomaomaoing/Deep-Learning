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

def load_data(domain):
#	 os.chdir(domain)
	categories = os.listdir(domain)
	label_num = dict(zip(categories, range(len(categories))))

	file = dict()
	for category in categories:
		temp = list(map(lambda x: domain+category+'/'+x, os.listdir(domain + category)))
		#temp = [t for t in temp if t[-3:] != 'png']
		file[category] = temp
	return file, label_num


def get_batch(data, categories, batch_size, batch_i, k):
	batch = []
	labels = []
	## get i-th batch
	for c in categories.keys():
		batch.extend(data[c][batch_i*int(batch_size/k):(batch_i+1)*int(batch_size/k)])
		labels += [categories[c]]*int(batch_size/k)
		
#	seed = random.randint(1,300)   
#	random.seed(seed)
#	random.shuffle(batch)
#	random.seed(seed)
#	random.shuffle(labels)
	
	return batch, labels

def shuffle(data):
	for c in data:
		random.shuffle(data[c])
	return data

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

class CNN_Model():
   
	def __init__(self, optm, lr, batch_size, k):
		
		
		self.lr = lr
		self.batch_size = batch_size
		self.optimizer = optm
		self.initial = tf.keras.initializers.he_normal()
		
		def preprocess(filename):
			"""name = tf.strings.substr(filename, -3, 3)
			png = tf.constant("png", dtype=tf.string)
			img = tf.cond(
					tf.equal(name, png), 
					lambda: tf.image.decode_png(tf.read_file(filename), 
												channels=4), 
					lambda: tf.image.decode_jpeg(tf.read_file(filename), 
												 channels=3))"""

			## read image
			img = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
			#img = tf.divide(tf.subtract(tf.cast(img, tf.float32), 127.5), 127.5)
			img = tf.reshape(tf.image.resize_images(img, [256, 256]), [256, 256, 3])
			"""img = tf.reshape(
				tf.image.resize_image_with_crop_or_pad(
					tf.image.resize_images(img, [256, 256], preserve_aspect_ratio = True)
					, 256, 256)
				, [256, 256, 3])"""

			"""img = tf.image.resize_image_with_crop_or_pad(
							tf.image.resize_images(img, [300, 300], preserve_aspect_ratio = True)
							, 300, 300)
							
			img = tf.cond(tf.equal(name, png), 
						  lambda: tf.cast(tf.reshape(img, [300, 300, 4]), tf.float32),
						  lambda: tf.reshape(
									  tf.concat([
											  tf.cast(img, tf.float32), 
											  tf.constant(1, shape=[300, 300, 1], dtype=tf.float32)], axis = 2), 
									  [300, 300, 4]))"""

			return img
			  
		self.x_path = tf.placeholder(tf.string, [None]) ## batch
		self.y = tf.placeholder(tf.int64, [None])
		
		self.imgs = tf.map_fn(preprocess, self.x_path, dtype = tf.float32)
		## normalization
#		self.imgs = tf.divide(tf.subtract(tf.cast(self.imgs, tf.float32), 127.5), 127.5)
		mean, var = tf.nn.moments(self.imgs, axes = [0, 1, 2])
		self.imgs = (self.imgs-mean) / tf.sqrt(var)
#		 print(self.imgs.shape)
		self.output = self.forward(self.imgs)
#		 print(self.output.shape)
		
		self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.y, k), self.output)
		gradients = self.optimizer.compute_gradients(self.loss)
		self.train_op = self.optimizer.apply_gradients(gradients)
		
		self.prediction = tf.argmax(self.output, 1)
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))
		
	def forward(self, input_img):
		
#		 print(input_img.shape)
		
		img = tf.layers.conv2d(input_img, 64, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		 
		img = tf.layers.conv2d(img, 64, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		
		img = tf.layers.max_pooling2d(img, 2, 2, 'same')
		
#		 print(img.shape)
		
		img = tf.layers.conv2d(img, 128, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		 
		img = tf.layers.conv2d(img, 128, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		
		img = tf.layers.max_pooling2d(img, 2, 2, 'same')
		
		img = tf.layers.conv2d(img, 256, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		 
		img = tf.layers.conv2d(img, 256, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		
		img = tf.layers.max_pooling2d(img, 2, 2, 'same')
		
		img = tf.layers.conv2d(img, 512, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		 
		img = tf.layers.conv2d(img, 512, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		
		img = tf.layers.max_pooling2d(img, 2, 2, 'same')
		
		img = tf.layers.conv2d(img, 512, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		 
		img = tf.layers.conv2d(img, 512, 3, padding = 'same', kernel_initializer = self.initial)
		img = tf.nn.relu(img)		
		img = tf.layers.max_pooling2d(img, 2, 2, 'same')
		
		print('before flatten', img.shape)
		
		img = tf.layers.flatten(img)
		
		print('flatten size', img.shape) # 8*8*512 = 32768
		img = tf.layers.dense(img, 2048, activation = "relu", kernel_initializer = self.initial, bias_initializer=tf.zeros_initializer())
		img = tf.layers.dense(img, 128, activation = "relu", kernel_initializer = self.initial, bias_initializer=tf.zeros_initializer())
		output = tf.layers.dense(img, 10, kernel_initializer = self.initial, bias_initializer=tf.zeros_initializer())
		
		return output


print("GPU available:", tf.test.is_gpu_available(), tf.test.gpu_device_name())




#%%

#widths = []
#heights = []
#  
#""" preprocessing """
### get all image's shape
#for category in categories: 
#	 for i, file in enumerate(train_file[category]):
#		 img = plt.imread(file, format=None)
#
#		 widths.append(img.shape[1])
#		 heights.append(img.shape[0])
#		 print(category, i)
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
k = 10 ## k = 10 categories
train_total = 10000
test_total = 400*k
batch_size = 100


n1_batch = int(train_total / batch_size)
n2_batch = int(test_total / batch_size)

# load data
#train_file, categories = load_data("D:\School\碩一\深度學習\DL_HW2\\animal-10\\train\\".replace('\\', '/'))
#test_file, _ = load_data("D:\School\碩一\深度學習\DL_HW2\\animal-10\\val\\".replace('\\', '/'))
#os.chdir(os.getcwd())
print("Now at", os.getcwd())
train_file, categories = load_data(os.getcwd().replace("\\", '/')+"/animal-10/train/")
test_file, _ = load_data(os.getcwd().replace("\\", '/')+"/animal-10/val/")
   
	
#%%

""" run """
lr = 5e-4
n_epoch = 100
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']= '0'

#config = tf.ConfigProto() 
#config.gpu_options.per_process_gpu_memory_fraction = 1
model = CNN_Model(optimizer, lr, batch_size, k)

train_acc, train_loss, test_acc, test_loss = [], [], [], []
saver = tf.train.Saver()

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	for epoch in range(n_epoch):
		
		train_file = shuffle(train_file)
		predict = []
		
		"""train"""
		for i in range(n1_batch):
			# get i-th batch
			batch, labels = get_batch(train_file, categories, batch_size, i, k)
			acc, _, loss = sess.run([model.accuracy, model.train_op, model.loss], 
									{model.x_path: batch, model.y:labels})
			if i % 10 ==0:
				print("Epoch %s, iter %s: train acc %s, loss %s" % (epoch, i, acc, loss))
		train_acc.append(acc); train_loss.append(loss)
		
		""" test """
		if epoch % 1 == 0:
			batch_acc = 0
			for i in range(n2_batch):
				# get i-th batch
				batch, labels = get_batch(test_file, categories, batch_size, i, k)
				acc, predict, loss = sess.run([model.accuracy, model.prediction, model.loss], 
										{model.x_path: batch, model.y:labels})
				batch_acc += acc
			print("Epoch %s: test acc %s"% (epoch, batch_acc/n2_batch))
			test_acc.append(acc); test_loss.append(loss)
	
	save_path = saver.save(sess, "model_k3fc16.ckpt")
	
""" save file """
save_json(train_acc, "train_acc_k3fc16.json")
save_json(train_loss, "train_loss_k3fc16.json")

save_json(test_acc, "test_acc_k3fc16.json")
save_json(test_loss, "test_loss_k3fc16.json")


#%%
""" test """

#import tensorflow as tf
#
#filename1 = "D:/School/碩一/深度學習/DL_HW2/animal-10/train/butterfly/e83cb80e29f11c22d2524518b7444f92e37fe5d404b0144390f8c770a3e5b7_640.jpg"
#filename2 = "D:/School/碩一/深度學習/DL_HW2/animal-10/train/butterfly/eb32b6092bf0033ed1584d05fb1d4e9fe777ead218ac104497f5c97faee8b1b8_640.png"
##x_path = tf.placeholder(tf.string, [1])
#
#
#
#name = tf.strings.substr(filename2, -3, 3)
#png = tf.constant("png", dtype=tf.string)
#img = tf.cond(
#		 tf.equal(name, png), 
#		 lambda: tf.image.decode_png(tf.read_file(filename2), channels=4), 
#		 lambda: tf.image.decode_jpeg(tf.read_file(filename2), channels=3))
#
#
##	  
##except:
##	  img = tf.image.decode_jpeg(tf.read_file(filename1), channels=3)
##img = tf.reshape(
##				  tf.image.resize_image_with_crop_or_pad(
##					  tf.image.resize_images(img, [300, 300], preserve_aspect_ratio = True)
##					  , 300, 300)
##				  , [300, 300, 3])
##		  
##img = tf.concat([tf.cast(img, tf.uint8), tf.constant(255, shape=[300, 300, 1], dtype=tf.uint8)], axis = 2)
##def preprocess(filename):
##	  print(filename.dtype)
##
##filetype = tf.map_fn(preprocess, x_path)
#
##path = x_path
#
#with tf.Session() as sess:
##	  img1 = sess.run([model.imgs], {model.x_path: train_file['cat']})
#	 img1 = sess.run(img)
# 
##plt.imshow(img1.astype('uint8'))
#

