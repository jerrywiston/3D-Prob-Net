import tensorflow as tf
import numpy as np
from ReadPc import *
from sklearn.neighbors import KDTree

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def uniform_samp(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def ProbNetSamp(kid, samp=200000, prob=0.94):
	x_re = uniform_samp(samp,3)
	k_re = KeySamp(samp, totalModel, kid)
	y_re = sess.run(y_reconst, feed_dict={x_: x_re, k_: k_re})
	pc_re = []
	for i in range(y_re.shape[0]):
		if y_re[i] > prob:
			pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

	return pc_re

def KeySamp(mb_size, totalModel, kid):
	k = [0.0]*totalModel
	k[kid] = 1.0
	k_samp = []
	for i in range(mb_size):
		k_samp.append(k)
	return k_samp

def ProbNetSamp2(key, samp=200000, prob=0.94):
	x_re = uniform_samp(samp,3)
	k_re = KeySamp2(key, samp, totalModel)
	y_re = sess.run(y_reconst, feed_dict={x_: x_re, k_: k_re})
	pc_re = []
	for i in range(y_re.shape[0]):
		if y_re[i] > prob:
			pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

	return pc_re

def KeySamp2(key, mb_size, totalModel):
	k_samp = []
	for i in range(mb_size):
		k_samp.append(key)
	return k_samp

#Probability Net
totalModel = 100
z_dim = 10

x_ = tf.placeholder(tf.float32, shape=[None, 3])
k_ = tf.placeholder(tf.float32, shape=[None, totalModel])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
z_ = tf.placeholder(tf.float32, shape=[None, z_dim])

W_z = tf.Variable(xavier_init([totalModel,z_dim]))

W1 = tf.Variable(xavier_init([3+z_dim,512]))
b1 = tf.Variable(tf.zeros(shape=[512]))

W2 = tf.Variable(xavier_init([512,800]))
b2 = tf.Variable(tf.zeros(shape=[800]))

W3 = tf.Variable(xavier_init([800,1024]))
b3 = tf.Variable(tf.zeros(shape=[1024]))

W4 = tf.Variable(xavier_init([1024,512]))
b4 = tf.Variable(tf.zeros(shape=[512]))

W5 = tf.Variable(xavier_init([512,1]))
b5 = tf.Variable(tf.zeros(shape=[1]))

def EncodeNet(k):
	z_digit = tf.matmul(k_, W_z)
	z = tf.nn.l2_normalize(z_digit, dim=1, epsilon=1, name=None)
	return z

def ProbNet(x, z):
	x_z = tf.concat(axis=1, values=[x, z])
	h1 = tf.nn.relu(tf.matmul(x_z, W1) + b1)
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
	h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
	h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
	y = tf.nn.sigmoid(tf.matmul(h4, W5) + b5)
	return y

z_encode = EncodeNet(k_)
y_reconst = ProbNet(x_, z_encode)
y_sample = ProbNet(x_, z_)

sess = tf.Session()
#sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "tf_save/chair_glo.ckpt")

'''
k = [0.0]*totalModel
k[4] = 0.5
k[23] = 0.5
pc_re = ProbNetSamp2(key = k)
print(len(pc_re))
DrawPc(pc_re,[[0,1],[0,1],[0,1]])

k = np.random.uniform(0., 1., size=[totalModel])
k /= np.sqrt(k.dot(k))
pc_re = ProbNetSamp2(key = k.tolist())
print(k)
print(len(pc_re))
DrawPc(pc_re,[[0,1],[0,1],[0,1]])
'''

for i in range(totalModel):
	pc_re = ProbNetSamp(kid = i)
	print(len(pc_re))
	DrawPc(pc_re,[[0,1],[0,1],[0,1]])