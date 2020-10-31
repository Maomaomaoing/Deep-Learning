# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:22:10 2019

@author: MaoChuLin
"""

import matplotlib.pyplot as plt
import json
import numpy as np

def load_json(filename):
	return json.load(open(filename, 'r'))

def moving_avg(loss, n):
	new_loss = []
	for i, l in enumerate(loss):
		if i<n:
			new_loss.append(l)
		elif i > len(loss)-n:
			break
		else:
			new_loss.append(sum(loss[i:i+n])/n)
	
	return new_loss

def plot_loss(DA_loss, DB_loss, G_loss, moving_n):
	plt.figure(figsize=(10, 6))
	plt.subplot(2,2,1)
	plt.title("Discriminator A loss")
	plt.plot(moving_avg(DA_loss, moving_n))
	plt.subplot(2,2,2)
	plt.title("Discriminator B loss")
	plt.plot(moving_avg(DB_loss, moving_n))
	plt.subplot(2,2,3)
	plt.title("Discriminator loss")
	plt.plot(moving_avg(np.array(DA_loss)+np.array(DB_loss), moving_n))
	plt.subplot(2,2,4)
	plt.title("Generator A loss")
	plt.plot(moving_avg(G_loss, moving_n))
	

#%%
#path = "output_061618/"
path = "output_071616/"
DA_loss = load_json(path+"DA_loss.json")
DB_loss = load_json(path+"DB_loss.json")
G_loss = load_json(path+"G_loss.json")

plot_loss(DA_loss, DB_loss, G_loss, 1000)