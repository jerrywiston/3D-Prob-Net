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
#sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "Model/chair.ckpt")

x_re = uniform_samp(500000,3)
y_re = sess.run(y,feed_dict={x_: x_re})
print(y_re.shape[0])

pc_re = []
for i in range(y_re.shape[0]):
	if y_re[i] > 0.94:
		pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

print(len(pc_re))
DrawPc(pc_re,[[0,1],[0,1],[0,1]])