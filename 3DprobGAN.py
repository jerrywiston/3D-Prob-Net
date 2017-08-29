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

def ProbNetSamp(z_dim, samp=500000, prob=0.94):
    x_re = uniform_samp(samp,3)
    z_re = OneNoiseSamp(samp, z_dim)
    y_re = sess.run(Gi_sample,feed_dict={xi_: x_re, zi_: z_re})
    pc_re = []
    for i in range(y_re.shape[0]):
        if y_re[i] > prob:
            pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

    return pc_re

z_dim = 100
s_dim = 128
b_size = 128

# Placeholder
xi_ = tf.placeholder(tf.float32, shape=[None, 3])
zi_ = tf.placeholder(tf.float32, shape=[None, z_dim])

x_ = tf.placeholder(tf.float32, shape=[b_size, s_dim, 3])
z_ = tf.placeholder(tf.float32, shape=[b_size, s_dim, z_dim])
y_ = tf.placeholder(tf.float32, shape=[b_size, s_dim, 1])

x_reshape_g = tf.reshape(x_, [b_size*s_dim, 3])
z_reshape_g = tf.reshape(z_, [b_size*s_dim, z_dim])

x_reshape_d = tf.reshape(x_, [b_size, s_dim*3])
y_reshape_d = tf.reshape(y_, [b_size, s_dim])

# Generator
W_g1 = tf.Variable(xavier_init([3+z_dim,256]))
b_g1 = tf.Variable(tf.zeros(shape=[256]))

W_g2 = tf.Variable(xavier_init([256,512]))
b_g2 = tf.Variable(tf.zeros(shape=[512]))

W_g3 = tf.Variable(xavier_init([512,256]))
b_g3 = tf.Variable(tf.zeros(shape=[256]))

W_g4 = tf.Variable(xavier_init([256,1]))
b_g4 = tf.Variable(tf.zeros(shape=[1]))

theta_G = [W_g1, b_g2, W_g2, b_g2, W_g3, b_g3, W_g4, b_g4]

def Generator(x, z):
    in_concat = tf.concat(axis=1, values=[x, z])

    h_g1 = tf.nn.relu(tf.matmul(in_concat, W_g1) + b_g1)
    h_g2 = tf.nn.relu(tf.matmul(h_g1, W_g2) + b_g2)
    h_g3 = tf.nn.relu(tf.matmul(h_g2, W_g3) + b_g3)
    
    y = tf.nn.sigmoid(tf.matmul(h_g3, W_g4) + b_g4)
    return y

# Discriminator
W_d1 = tf.Variable(xavier_init([s_dim*4, 512]))
b_d1 = tf.Variable(tf.zeros(shape=[512]))

W_d2 = tf.Variable(xavier_init([512, 256]))
b_d2 = tf.Variable(tf.zeros(shape=[256]))

W_d3 = tf.Variable(xavier_init([256, 128]))
b_d3 = tf.Variable(tf.zeros(shape=[128]))

W_d4 = tf.Variable(xavier_init([128, 1]))
b_d4 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [W_d1, b_d2, W_d2, b_d2, W_d3, b_d3, W_d4, b_d4]

def Discriminator(x, y):
    in_concat = tf.concat(axis=1, values=[x, y])
    
    h_d1 = tf.nn.relu(tf.matmul(in_concat, W_d1) + b_d1)
    h_d2 = tf.nn.relu(tf.matmul(h_d1, W_d2) + b_d2)
    h_d3 = tf.nn.relu(tf.matmul(h_d2, W_d3) + b_d3)
    
    D_logit = tf.matmul(h_d3, W_d4) + b_d4
    D_prob = tf.nn.sigmoid(D_logit)
    
    return D_logit

Gi_sample = Generator(xi_, zi_)
G_sample = Generator(x_reshape_g, z_reshape_g)
G_sample_reshape = tf.reshape(G_sample,[b_size, s_dim])

D_real = Discriminator(x_reshape_d, y_reshape_d)
D_fake = Discriminator(x_reshape_d, G_sample_reshape)

D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean((D_fake + 1)**2))
G_loss = 0.5 * tf.reduce_mean((D_fake)**2)

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def OneRealSamp(pc, tree, dim, rate):
    x_samp = []
    y_samp = []
    for i in range(dim):
        r = np.random.uniform(0., 1., size=[1])[0]
        if r > rate:
            x_uni_samp = uniform_samp(1,3)
            y_uni_samp = prob_samp(tree, x_uni_samp)
            x_samp.extend(x_uni_samp.tolist())
            y_samp.extend(y_uni_samp.tolist())
        else:
            x_pc_samp, y_pc_samp = next_batch(pc, 1)
            x_samp.extend(x_pc_samp.tolist())
            y_samp.extend(y_pc_samp.tolist())

    return x_samp, y_samp

def OneNoiseSamp(s_dim, z_dim):
    z = np.random.uniform(-1., 1., size=[z_dim]).tolist()
    z_ = []
    for i in range(s_dim):
        z_.append(z)
    return z_

def BatchSamp(train_pc, train_tree, b_size, s_dim, z_dim, rate):
    x = []
    y_real = []
    z_samp = []
    for i in range(b_size):
        r = random.randint(0,len(train_pc)-1)
        
        x_, y_ = OneRealSamp(train_pc[r], train_tree[r], s_dim, rate)
        x.append(x_)
        y_real.append(y_)

        z_ = OneNoiseSamp(s_dim, z_dim)
        z_samp.append(z_)

    return x, y_real, z_samp

# Read npts
path = "3d_model/ModelNet10_chair/"
files = [f for f in os.listdir(path)]

train_pc = []
for filename in files:
    f = filename.split(".")
    if len(f)==2 and f[1].strip()=="npts":
        pc = TrainSampNpts(path + filename)
        train_pc.append(pc)

# Build kd-tree
train_tree = []
for pc in train_pc:
    tree = KDTree(pc, leaf_size=2)
    train_tree.append(tree)


for i in range(100000):
    x, y_real, z_samp = BatchSamp(train_pc, train_tree, b_size, s_dim, z_dim, 0.5)
    x_np = np.asarray(x)
    y_np = np.asarray(y_real)
    z_np = np.asarray(z_samp)
    #print(x_np.shape)
    #print(y_np.shape)
    #print(z_np.shape)
    #print(y_real)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x_:x_np, y_:y_np, z_:z_np})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x_:np.asarray(x), z_:np.asarray(z_samp)})

    if i%10 == 0:
        print(str(i) + " " + str(D_loss_curr) + " " + str(G_loss_curr))
        pc_re = ProbNetSamp(z_dim, samp=50000, prob=0.5)
        DrawPc(pc_re, show=False, filename="out/" + str(i))
