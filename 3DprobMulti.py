import tensorflow as tf
import numpy as np
from ReadPc import *
from sklearn.neighbors import KDTree
import random
import os

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def uniform_samp(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def prob_samp(tree, x_samp):
	dist, _ = tree.query(x_samp, k=1)
	res = 0.05
	#print(dist.shape)
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

def ProbNetSamp(totalModel, samp=500000, prob=0.94):
	x_re = uniform_samp(samp,3)
	kid = random.randint(0, totalModel-1)
	print("Samp id: " + str(kid+1))
	k_re = KeySamp(samp, totalModel, kid)
	y_re = sess.run(y,feed_dict={x_: x_re, k_: k_re})
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

#Probability Net
totalModel = 25
x_ = tf.placeholder(tf.float32, shape=[None, 3])
k_ = tf.placeholder(tf.float32, shape=[None, totalModel])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(xavier_init([3+totalModel,256]))
b1 = tf.Variable(tf.zeros(shape=[256]))

W2 = tf.Variable(xavier_init([256,512]))
b2 = tf.Variable(tf.zeros(shape=[512]))

W3 = tf.Variable(xavier_init([512,700]))
b3 = tf.Variable(tf.zeros(shape=[700]))

W4 = tf.Variable(xavier_init([700,256]))
b4 = tf.Variable(tf.zeros(shape=[256]))

W5 = tf.Variable(xavier_init([256,1]))
b5 = tf.Variable(tf.zeros(shape=[1]))

x_k = tf.concat(axis=1, values=[x_, k_])
h1 = tf.nn.relu(tf.matmul(x_k, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
y = tf.nn.sigmoid(tf.matmul(h4, W5) + b5)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
solver = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

path = "3d_model/ModelNet10_chair/"
files = [f for f in os.listdir(path)]
pc_list = []
tree_list = []
count = 0
for filename in files:
    f = filename.split(".")
    if len(f)==2 and f[1].strip()=="npts":
        pc = TrainSampNpts(path + filename)
        pc_list.append(pc)
        tree = KDTree(pc, leaf_size=2)
        tree_list.append(tree)
        
        count += 1
        if count >= totalModel:
        	break

mb_size = 200
for i in range(500000):
	kid = random.randint(0,totalModel-1)
	x_uni_samp = uniform_samp(mb_size,3)
	y_uni_samp = prob_samp(tree_list[kid], x_uni_samp)
	x_pc_samp, y_pc_samp = next_batch(pc_list[kid], round(mb_size/2))

	x_samp = np.concatenate((x_uni_samp, x_pc_samp), axis=0)
	y_samp = np.concatenate((y_uni_samp, y_pc_samp), axis=0)
	k_samp = np.asarray(KeySamp(x_samp.shape[0], totalModel, kid))

	_, _loss = sess.run([solver, loss],  feed_dict={x_: x_samp, k_: k_samp, y_: y_samp})

	if i%1000 == 0:
		print(str(i) + " " + str(_loss))
		pc_re = ProbNetSamp(totalModel, samp=50000, prob=0.9)
		DrawPc(pc_re, show=False, filename="out/" + str(i))

print("Save parameter ...")
saver = tf.train.Saver()
save_path = saver.save(sess, "tf_save/chair.ckpt")