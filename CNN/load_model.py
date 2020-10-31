# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:34:24 2019

@author: MaoChuLin
"""
import os
import tensorflow as tf
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
	
	return batch, labels
def save_json(data, filename):
	to_float = lambda num: float(num)
	with open(filename, 'w') as f:
		json.dump(list(map(to_float, data)), f)
	print("Save file at", filename)

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
		img = tf.layers.dense(img, 4096, activation = "relu", kernel_initializer = self.initial, bias_initializer=tf.zeros_initializer())
		img = tf.layers.dense(img, 512, activation = "relu", kernel_initializer = self.initial, bias_initializer=tf.zeros_initializer())
		img = tf.layers.dense(img, 64, activation = "relu", kernel_initializer = self.initial, bias_initializer=tf.zeros_initializer())
		output = tf.layers.dense(img, 10, kernel_initializer = self.initial, bias_initializer=tf.zeros_initializer())
		
		return output
    
  
k = 10
lr = 5e-4
batch_size = 100
test_total = 4000
optimizer = tf.train.AdamOptimizer(learning_rate = lr)

tf.reset_default_graph()  
model = CNN_Model(optimizer, lr, batch_size, k)

_, categories = load_data(os.getcwd().replace("\\", '/')+'/..'+"/animal-10/train/")
test_file, _ = load_data(os.getcwd().replace("\\", '/')+'/..'+"/animal-10/val/")
n2_batch = int(test_total / batch_size)

saver = tf.train.Saver()

#tf.reset_default_graph()
with tf.Session() as sess:
#	saver = tf.train.import_meta_graph('model_k3.ckpt.meta')
#	saver.restore(sess, tf.train.latest_checkpoint('./'))
#    sess.run(tf.global_variables_initializer())
	saver.restore(sess, 'model_k3.ckpt')
	
	"""all_vars = tf.trainable_variables()
	for v in all_vars:
		print("%s with value %s" % (v.name, sess.run(v)))"""
	
	
	batch_acc = 0
	for i in range(n2_batch):
		# get i-th batch
		batch, labels = get_batch(test_file, categories, batch_size, i, k)
		acc, predict, loss = sess.run([model.accuracy, model.prediction, model.loss], 
										{model.x_path: batch, model.y:labels})
		batch_acc += acc
	print(batch_acc/n2_batch)
#    sess.run(tf.global_variables_initializer())
""" save file """
save_json(predict, "predict.json")
