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

def ProbNetSamp(key, samp=200000, prob=0.94):
	x_re = uniform_samp(samp,3)
	k_re = KeyTensor(key, samp)
	y_re = sess.run(y_reconst, feed_dict={x_: x_re, k_: k_re})
	pc_re = []
	for i in range(y_re.shape[0]):
		if y_re[i] > prob:
			pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

	return pc_re

def VoxelX(vox_len):
	x_re = []
	for i in range(vox_len):
		for j in range(vox_len):
			for k in range(vox_len):
				x_re.append([float(i)/vox_len, float(j)/vox_len, float(k)/vox_len])
	return x_re

def ProbNetSampVoxel(key, vox_len=64, prob=0.94):
	x_re = VoxelX(vox_len)

	k_re = KeyTensor(key, len(x_re))
	y_re = sess.run(y_reconst, feed_dict={x_: x_re, k_: k_re})
	pc_re = []
	for i in range(y_re.shape[0]):
		if y_re[i] > prob:
			pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

	return pc_re


def KeyTensor(key, mb_size):
	k_samp = []
	for i in range(mb_size):
		k_samp.append(key)
	return k_samp

def ProbNetIdSamp(kid, samp=200000, prob=0.94):
	key = OneKeyId(kid, totalModel)
	pc_re = ProbNetSamp(key, samp, prob)

	return pc_re

def ProbNetIdSampVoxel(kid, vox_len=64, prob=0.94):
	key = OneKeyId(kid, totalModel)
	pc_re = ProbNetSampVoxel(key, vox_len, prob)

	return pc_re

def OneKeyId(kid, totalModel):
	k = [0.0]*totalModel
	k[kid] = 1.0
	return k

def OneKeyRand(totalModel):
	k = np.random.uniform(0., 1., size=[totalModel])
	k /= np.sqrt(k.dot(k))
	return k.tolist()

def OneZRand(z_dim):
	scale = random.uniform(0., 1.)
	z_np = np.random.uniform(-1., 1., size=[z_dim])
	z_np /= np.sqrt(z_np.dot(z_np))
	z_np *= scale
	z = z_np.tolist()
	return z

def ZSamp(z, samp=100000, prob=0.9):
	print("Latent Variable: ")
	print(z)
	z_samp = []

	for i in range(samp):
		z_samp.append(z)
	
	pc_samp = []

	x_samp = uniform_samp(samp,3)
	y_samp = sess.run(y_sample, feed_dict={x_: x_samp, z_: z_samp})

	for i in range(y_samp.shape[0]):
		if y_samp[i] > prob:
			pc_samp.append([x_samp[i][0], x_samp[i][1], x_samp[i][2]])

	return pc_samp

def ZSampVoxel(z, vox_len=50, prob=0.9):
	print("Latent Variable: ")
	print(z)
	z_samp = []

	for i in range(vox_len*vox_len*vox_len):
		z_samp.append(z)
	
	pc_samp = []

	x_samp = VoxelX(vox_len)
	y_samp = sess.run(y_sample, feed_dict={x_: x_samp, z_: z_samp})

	for i in range(y_samp.shape[0]):
		if y_samp[i] > prob:
			pc_samp.append([x_samp[i][0], x_samp[i][1], x_samp[i][2]])

	return pc_samp

def Znormalize(z):
	z_np = np.asarray(z)
	z_np /= np.sqrt(z_np.dot(z_np))
	return z_np.tolist()

#Probability Net
totalModel = 250
z_dim = 10
v_size = 32
mb_size = 200

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
saver.restore(sess, "tf_save/chair_glo_889.ckpt")

def ViewToTensor(v, mb_size):
	v_ts = []
	for i in range(mb_size):
		v_ts.append(v)
	return v_ts

#Conv Net
W_conv1 = tf.Variable(xavier_init([5,5,3,16]))
b_conv1 = tf.zeros(shape=[16])

W_conv2 = tf.Variable(xavier_init([3,3,16,16]))
b_conv2 = tf.zeros(shape=[16])

W_fc1 = tf.Variable(xavier_init([8*8*16,128]))
b_fc1 = tf.zeros(shape=[128])

W_fc2 = tf.Variable(xavier_init([128,z_dim]))
b_fc2 = tf.zeros(shape=[z_dim])

v_f = tf.placeholder(tf.float32, shape=[None, v_size, v_size, 3])
z_f = tf.placeholder(tf.float32, shape=[None, z_dim])

def LoadDataId(path, totalModel):
	pc_list = []
	tree_list = []
	view_list = []
	count = 0

	for i in range(totalModel):
		filename = str(i) + ".npts"
		pc = TrainSampNpts(path + filename)
		pc_list.append(pc)
		tree = KDTree(pc, leaf_size=2)
		tree_list.append(tree)
		view_list.append(NptsToView(pc, 32))

	return pc_list, tree_list, view_list

def next_batch(imgs, labels, size):
    img_samp = np.ndarray(shape=(size, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    label_samp = np.ndarray(shape=(size, labels.shape[1]))
    for i in range(size):
        rand_num = random.randint(0,len(imgs)-1)
        img_samp[i] = imgs[rand_num]
        label_samp[i] = labels[rand_num]
    return [img_samp, label_samp]

def kl_divergence(p, q): 
    return tf.reduce_sum(p * tf.log(p/q))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def FeatureNet(v):
	h_conv1 = tf.nn.relu(conv2d(v, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*16])
	
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

	return h_fc2

sess2 = tf.Session()
sess2.run(tf.global_variables_initializer())
z_feature = FeatureNet(v_f)

loss = kl_divergence(z_feature, z_f)
solver = tf.train.AdamOptimizer().minimize(loss)

### z samp test
z = sess.run(W_z)
path = "3d_model/ModelNet10_chair/"
[pc_list, tree_list, view_list] = LoadDataId(path, 10)

for i in range(400000):
	v_samp, z_samp = next_batch(np.asarray(view_list), z,100)
	_, _loss = sess.run([solver, loss],  feed_dict={v_f: v_samp, z_f: z_samp})
	if i%100 == 0:
		print(str(i) + " " + str(_loss))