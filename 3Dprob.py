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

#Probability Net
x_ = tf.placeholder(tf.float32, shape=[None, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(xavier_init([3,256]))
b1 = tf.Variable(tf.zeros(shape=[256]))

W2 = tf.Variable(xavier_init([256,512]))
b2 = tf.Variable(tf.zeros(shape=[512]))

W3 = tf.Variable(xavier_init([512,256]))
b3 = tf.Variable(tf.zeros(shape=[256]))

W4 = tf.Variable(xavier_init([256,1]))
b4 = tf.Variable(tf.zeros(shape=[1]))

h1 = tf.nn.relu(tf.matmul(x_, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
y = tf.nn.sigmoid(tf.matmul(h3, W4) + b4)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
solver = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Read Pointcloud
pc = ReadNpts("horse.npts", -1)
Rescale(pc)
#DrawPc(pc,[[0,1],[0,1],[0,1]])
tree = KDTree(pc, leaf_size=2)

mb_size = 100
for i in range(100000):
	x_uni_samp = uniform_samp(mb_size,3)
	y_uni_samp = prob_samp(tree, x_uni_samp)
	x_pc_samp, y_pc_samp = next_batch(pc, round(mb_size/2))

	x_samp = np.concatenate((x_uni_samp, x_pc_samp), axis=0)
	y_samp = np.concatenate((y_uni_samp, y_pc_samp), axis=0)

	_, _loss = sess.run([solver, loss],  feed_dict={x_: x_samp, y_: y_samp})
	if i%1000 == 0:
		print(str(i) + " " + str(_loss))

print("Save parameter ...")
saver = tf.train.Saver()
save_path = saver.save(sess, "Model/model.ckpt")

'''
x_re = uniform_samp(100000,3)
y_re = sess.run(y,feed_dict={x_: x_re})
print(y_re.shape[0])

pc_re = []
for i in range(y_re.shape[0]):
	if y_re[i] > 0.8:
		pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

print(len(pc_re))
#print(pc_re)
DrawPc(pc_re,[[0,1],[0,1],[0,1]])
'''