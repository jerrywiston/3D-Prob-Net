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

def normal_init(size):
    return tf.random_normal(shape=size, stddev=0.1)

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

#Probability Net
totalModel = 20
z_dim = 8
mb_size = 200

x_ = tf.placeholder(tf.float32, shape=[None, 3])
k_ = tf.placeholder(tf.float32, shape=[None, totalModel])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
z_ = tf.placeholder(tf.float32, shape=[None, z_dim])

#Encoder
W_e1 = tf.Variable(normal_init([totalModel, 32]))
b_e1 = tf.Variable(tf.zeros(shape=[32]))

W_e2_mu = tf.Variable(normal_init([32, z_dim]))
b_e2_mu = tf.Variable(tf.zeros(shape=[z_dim]))
W_e2_sigma = tf.Variable(normal_init([32, z_dim]))
b_e2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))

var_e = [W_e1, b_e1, W_e2_mu, b_e2_mu, W_e2_sigma, b_e2_sigma]

def EncodeNet(k):
	h_e1 = tf.nn.relu(tf.matmul(k, W_e1) + b_e1)
	z_mu = tf.matmul(h_e1, W_e2_mu) + b_e2_mu
	z_logvar = tf.matmul(h_e1, W_e2_sigma) + b_e2_sigma
	return z_mu, z_logvar

#Genarator
W_p1 = tf.Variable(xavier_init([3+z_dim,512]))
b_p1 = tf.Variable(tf.zeros(shape=[512]))

W_p2 = tf.Variable(xavier_init([512,800]))
b_p2 = tf.Variable(tf.zeros(shape=[800]))

W_p3 = tf.Variable(xavier_init([800,1024]))
b_p3 = tf.Variable(tf.zeros(shape=[1024]))

W_p4 = tf.Variable(xavier_init([1024,512]))
b_p4 = tf.Variable(tf.zeros(shape=[512]))

W_p5 = tf.Variable(xavier_init([512,1]))
b_p5 = tf.Variable(tf.zeros(shape=[1]))

var_p = [W_p1, b_p1, W_p2, b_p2, W_p3, b_p3, W_p4, b_p4, W_p5, b_p5]

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

def ProbNet(x, z):
	z_re = tf.nn.l2_normalize(z, dim=1, epsilon=1, name=None)
	x_z = tf.concat(axis=1, values=[x, z_re])
	h_p1 = tf.nn.relu(tf.matmul(x_z, W_p1) + b_p1)
	h_p2 = tf.nn.relu(tf.matmul(h_p1, W_p2) + b_p2)
	h_p3 = tf.nn.relu(tf.matmul(h_p2, W_p3) + b_p3)
	h_p4 = tf.nn.relu(tf.matmul(h_p3, W_p4) + b_p4)
	y = tf.nn.sigmoid(tf.matmul(h_p4, W_p5) + b_p5)
	return y

# Load Data
def LoadData(path, totalModel):
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

	return pc_list, tree_list

def LoadDataId(path, totalModel):
	pc_list = []
	tree_list = []
	count = 0

	for i in range(totalModel):
		filename = str(i) + ".npts"
		pc = TrainSampNpts(path + filename)
		pc_list.append(pc)
		tree = KDTree(pc, leaf_size=2)
		tree_list.append(tree)

	return pc_list, tree_list


z_mu, z_logvar = EncodeNet(k_)
z_encode = sample_z(z_mu, z_logvar)
y_reconst = ProbNet(x_, z_encode)
y_sample = ProbNet(x_, z_)

#Loss and optimizer
re_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_reconst - y_), reduction_indices=[1]))
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
loss = tf.reduce_mean(re_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

path = "3d_model/ModelNet10_chair/"
[pc_list, tree_list] = LoadDataId(path, totalModel)

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
		if i%10000 == 0 and i>5000:
			pc_re = ProbNetSamp(totalModel, samp=50000, prob=0.9)
			print("Size: " + str(len(pc_re)))
			DrawPc(pc_re, show=False, filename="out/" + str(i))


print("Save parameter ...")
saver = tf.train.Saver()
save_path = saver.save(sess, "tf_save/chair_gglo.ckpt")