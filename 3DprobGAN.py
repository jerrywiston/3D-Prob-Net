import tensorflow as tf
import numpy as np
from ReadPc import *
from sklearn.neighbors import KDTree
import random

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def uniform_samp(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def prob_samp(tree, x_samp):
	dist, _ = tree.query(x_samp, k=1)
	res = 0.05
	y = np.zeros(shape=dist.shape)
	for i in range(len(dist)):
		if(dist[i]<res):
			y[i] = (res - dist[i]) / res
	return y


def next_batch(imgs, size):
    img_samp = np.ndarray(shape=(size, 3))
    label_samp = np.ndarray(shape=(size, 1))
    for i in range(size):
        rand_num = random.randint(0,len(imgs)-1)
        img_samp[i] = imgs[rand_num]
        label_samp[i] = 1.0
    return [img_samp, label_samp]

z_dim = 100
samp_size = 512
mb_size = 100

# Placeholder
x_ = tf.placeholder(tf.float32, shape=[None, samp_size, 3])
z_ = tf.placeholder(tf.float32, shape=[None, samp_size, z_dim])
y_ = tf.placeholder(tf.float32, shape=[None, samp_size, 1])

x_samp = tf.placeholder(tf.float32, shape=[None, 3])

# Generator
W_g1 = tf.Variable(xavier_init([3+z_dim,256]))
b_g1 = tf.Variable(tf.zeros(shape=[256]))

W_g2 = tf.Variable(xavier_init([256,512]))
b_g2 = tf.Variable(tf.zeros(shape=[512]))

W_g3 = tf.Variable(xavier_init([512,256]))
b_g3 = tf.Variable(tf.zeros(shape=[256]))

W_g4 = tf.Variable(xavier_init([256,1]))
b_g4 = tf.Variable(tf.zeros(shape=[1]))

def Generator(x, z):
    x_reshape = tf.reshape(x, [None, 3])
    z_reshape = tf.reshape(z, [None, z_dim])
    in_concat = tf.concat(axis=1, values=[x_reshape, z_reshape])

    h_g1 = tf.nn.relu(tf.matmul(in_concat, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    h_g3 = tf.nn.relu(tf.matmul(h_g2, W_g3) + b_g3)
    
    y = tf.nn.sigmoid(tf.matmul(h_g3, W_g4) + b_g4)
    return y

# Discriminator
W_d1 = tf.Variable(xavier_init([samp_size*4, 512]))
b_d1 = tf.Variable(tf.zeros(shape=[512]))

W_d2 = tf.Variable(xavier_init([512, 256]))
b_d2 = tf.Variable(tf.zeros(shape=[256]))

W_d3 = tf.Variable(xavier_init([256, 128]))
b_d3 = tf.Variable(tf.zeros(shape=[128]))

W_d4 = tf.Variable(xavier_init([128, 1]))
b_d4 = tf.Variable(tf.zeros(shape=[1]))

def Discriminator(x, y):
    x_reshape = tf.reshape(x, [None, samp_size*3])
    y_reshape = tf.reshape(y, [None, samp_size*1])
    in_concat = tf.concat(axis=1, values=[x_reshape, y_reshape])
    
    h_d1 = tf.nn.relu(tf.matmul(in_concat, W_g1) + b_g1)
    h_d2 = tf.nn.relu(tf.matmul(h_d1, W_d2) + b_d2)
    h_d3 = tf.nn.relu(tf.matmul(h_d2, W_d3) + b_d3)
    
    D_logit = tf.matmul(h_d3, W_d4) + b_d4
    D_prob = tf.nn.sigmoid(D_logit)
    
    return D_logit

D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean((D_fake + 1)**2))
G_loss = 0.5 * tf.reduce_mean((D_fake)**2)

