import tensorflow as tf
import numpy as np
import math
from ReadPc import *
from sklearn.neighbors import KDTree
import random
import os
from datetime import datetime

#========================= Util Function =========================
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def normal_init(size):
    return tf.random_normal(shape=size, stddev=0.1)

def uniform_samp(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def gaussian(x, u, sig):
	return np.exp(-(x - u) ** 2 /(2* sig **2))

def prob_samp(tree, x_samp, f=1):
	dist, _ = tree.query(x_samp, k=1)
	if f==0:	
		res = 0.05
		y = np.zeros(shape=dist.shape)
		for i in range(len(dist)):
			if(dist[i]<res):
				y[i] = (res - dist[i]) / res
	else:
		y = gaussian(dist, 0.0, 0.03).tolist()
	return y

def next_batch(imgs, size):
    img_samp = np.ndarray(shape=(size, 3))
    label_samp = np.ndarray(shape=(size, 1))
    for i in range(size):
        rand_num = random.randint(0,len(imgs)-1)
        img_samp[i] = imgs[rand_num]
        label_samp[i] = 1.0
    return [img_samp, label_samp]

def ProbNetSamp(key, samp=200000, prob=0.94):
	x_re = uniform_samp(samp,3)
	k_re = KeyTensor(key, samp)
	y_re = sess.run(y_reconst, feed_dict={x_: x_re, k_: k_re})
	pc_re = []
	for i in range(y_re.shape[0]):
		if y_re[i] > prob:
			pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

	return pc_re

def ProbNetSampVoxel(key, vox_len=64, prob=0.94):
	x_re = []
	samp = vox_len ** 3
	for i in range(vox_len):
		for j in range(vox_len):
			for k in range(vox_len):
				x_re.append([float(i)/vox_len, float(j)/vox_len, float(k)/vox_len])

	k_re = KeyTensor(key, samp)
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
	z_np = np.random.uniform(-1., 1., size=[z_dim])
	z_np /= np.sqrt(z_np.dot(z_np))
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

def Znormalize(z):
	z_np = np.asarray(z)
	z_np /= np.sqrt(z_np.dot(z_np))
	return z_np.tolist()

def LatentRescale(z):
    for i in range(z.shape[0]):
        length = z[i].dot(z[i])
        if length > 1.0:
            z[i] /= length
    return z

# Training data handle
def LoadDataFile(path, totalModel):
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

def LoadDataFileId(path, totalModel):
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

def LoadDataOneFile(path, id):
	filename = str(i) + ".npts"
	pc = TrainSampNpts(path + filename)
	tree = KDTree(pc, leaf_size=2)

	return pc, tree

def NextTrainBatch(totalModel, mb_size, kid, pc, tree):
	mb_uni_size = int(mb_size*2/3)
	mb_pc_size = mb_size - mb_uni_size

	x_uni_samp = uniform_samp(mb_uni_size,3)
	y_uni_samp = prob_samp(tree, x_uni_samp)
	x_pc_samp, y_pc_samp = next_batch(pc, mb_pc_size)

	x_samp = np.concatenate((x_uni_samp, x_pc_samp), axis=0)
	y_samp = np.concatenate((y_uni_samp, y_pc_samp), axis=0)
	k_samp = np.asarray(KeySamp(x_samp.shape[0], totalModel, kid))

	return x_samp, y_samp, k_samp

def WriteMessage(file, msg):
	print(msg)
	file.writelines(msg + "\n")

#========================= Net Model =========================
totalModel = 200
z_dim = 16
mb_size = 300
v_size = 32

v_ = tf.placeholder(tf.float32, shape=[None, v_size, v_size, 3])
x_ = tf.placeholder(tf.float32, shape=[None, 3])
k_ = tf.placeholder(tf.float32, shape=[None, totalModel])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
z_ = tf.placeholder(tf.float32, shape=[None, z_dim])
z_re = tf.placeholder(tf.float32, shape=[None, z_dim])

#Conv Net
W_conv1 = tf.Variable(xavier_init([5,5,3,16]))
b_conv1 = tf.zeros(shape=[16])

W_conv2 = tf.Variable(xavier_init([3,3,16,16]))
b_conv2 = tf.zeros(shape=[16])

W_fc1 = tf.Variable(xavier_init([8*8*16,128]))
b_fc1 = tf.zeros(shape=[128])

W_fc2 = tf.Variable(xavier_init([128,z_dim]))
b_fc2 = tf.zeros(shape=[z_dim])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def kl_divergence(p, q): 
    return tf.reduce_sum(p * tf.log(p/q))

def FeatureNet(v):
	h_conv1 = tf.nn.relu(conv2d(v, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*16])
	
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

	return h_fc2

# Probability Net
W_z = tf.Variable(normal_init([totalModel,z_dim]))

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
	z_digit = tf.matmul(k, W_z)
	#z = tf.nn.l2_normalize(z_digit, dim=1, epsilon=1, name=None)
	return z_digit

def ProbNet(x, z):
	x_z = tf.concat(axis=1, values=[x, z])
	h1 = tf.nn.relu(tf.matmul(x_z, W1) + b1)
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
	h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
	h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
	y = tf.nn.sigmoid(tf.matmul(h4, W5) + b5)
	return y

# Build Net
z_feature = FeatureNet(v_)
z_encode = EncodeNet(k_)
z_rescale = tf.assign(W_z, z_re)
y_reconst = ProbNet(x_, z_encode)
y_sample = ProbNet(x_, z_)

latent_loss = kl_divergence(z_feature, z_encode)
recon_loss = tf.nn.l2_loss(y_reconst - y_)
loss = latent_loss + recon_loss
solver = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#========================= Training =========================

saveName = "chair_glo_250.ckpt"
path = "3d_model/ModelNet10_chair/"
[pc_list, tree_list] = LoadDataFileId(path, totalModel)

# Record file
file = open("out/Record.txt", "a")
msg = "(" + saveName + ")"
WriteMessage(file, msg)

for i in range(400001):
	# Optimize
	#kid = random.randint(0,totalModel-1)
	kid = i % totalModel
	x_samp, y_samp, k_samp = NextTrainBatch(totalModel, mb_size, kid, pc_list[kid], tree_list[kid])
	_, _loss = sess.run([solver, loss],  feed_dict={x_: x_samp, k_: k_samp, y_: y_samp})

	# Rescale
	z_latent = sess.run(W_z)
	z_latent = LatentRescale(z_latent)
	sess.run(z_rescale, feed_dict={z_re: z_latent})

	# Print msg
	if i%10000 == 0:
		pc_re = ProbNetIdSampVoxel(kid, 50, 0.94)
		msg = "[" + str(datetime.now()) + "] Sample id " + str(kid+1) + ", Size = " + str(len(pc_re))
		WriteMessage(file, msg)
		DrawPc(pc_re, show=False, filename="out/" + str(i))

	if i%1000 == 0:
		msg = str(i) + " Iter, Loss: " + str(_loss)
		WriteMessage(file, msg)

file.close()
print("Save parameter ...")
saver = tf.train.Saver()
save_path = saver.save(sess, "tf_save/" + saveName)