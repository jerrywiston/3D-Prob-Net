import tensorflow as tf
import numpy as np
import math
from ReadPc import *
from sklearn.neighbors import KDTree
import random
import os
from datetime import datetime

#========================= Util Function =========================
# File Load
def LoadDataOneFileId(path, id):
	filename = str(id) + ".npts"
	pc = TrainSampNpts(path + filename)
	tree = KDTree(pc, leaf_size=2)
	return pc, tree

def LoadDataFileId(path, totalModel):
	pc_list = []
	tree_list = []

	for i in range(totalModel):
		pc, tree = LoadDataOneFileId(path, i)
		pc_list.append(pc)
		tree_list.append(tree)

	return pc_list, tree_list

def LoadDataFileFolder(path, totalModel):
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

#Train Sample Generate
def Gaussian(x, u, sig):
	return np.exp(-(x - u) ** 2 /(2* sig **2))

def UniformSamp(m, n):
    return np.random.uniform(0., 1., size=[m, n])

def TrainSpaceSamp(tree, x_samp):
	dist, _ = tree.query(x_samp, k=1)
	y = Gaussian(dist, 0.0, 0.04).tolist()
	return y

def TrainPcSamp(pc, size):
    pc_samp = np.ndarray(shape=(size, 3))
    label_samp = np.ndarray(shape=(size, 1))
    for i in range(size):
        r = random.randint(0,len(pc)-1)
        pc_samp[i] = pc[r]
        label_samp[i] = 1.0
    return [pc_samp, label_samp]

def NextTrainBatch(totalModel, mb_size, pc, tree):
	mb_uni_size = int(mb_size*2/3)
	mb_pc_size = mb_size - mb_uni_size

	x_uni_samp = UniformSamp(mb_uni_size,3)
	y_uni_samp = TrainSpaceSamp(tree, x_uni_samp)
	x_pc_samp, y_pc_samp = TrainPcSamp(pc, mb_pc_size)

	x_samp = np.concatenate((x_uni_samp, x_pc_samp), axis=0)
	y_samp = np.concatenate((y_uni_samp, y_pc_samp), axis=0)

	return x_samp, y_samp

#ProbNet Sample
def ZTensor(z, mb_size):
	z_samp = np.ndarray(shape=(mb_size, z.shape[0]))
	for i in range(mb_size):
		z_samp[i] = z
	return z_samp

def ZSamp(latent_size):
	z_np = np.random.normal(0., 0.3, [latent_size])
	return z_np

def ProbNetSamp(sess, z, samp, x_re, prob):
	z_re = ZTensor(z, samp)
	y_re = sess.run(y_prob, feed_dict={x_: x_re, k_: z_re})
	pc_re = []
	for i in range(y_re.shape[0]):
		if y_re[i] > prob:
			pc_re.append([x_re[i][0], x_re[i][1], x_re[i][2]])

	return pc_re

def ProbNetSampRand(sess, z, samp=200000, prob=0.94):
	x_re = UniformSamp(samp,3)
	pc_re = ProbNetSamp(sess, z, samp, x_re, prob)
	return pc_re

def ProbNetSampVoxel(sess, z, vox_len=50, prob=0.94):
	total = int(vox_len ** 3)
	x_re = []
	count = 0
	for i in range(int(vox_len)):
		for j in range(int(vox_len)):
			for k in range(int(vox_len)):
				x_re.append([float(i)/vox_len, float(j)/vox_len, float(k)/vox_len])
				count += 1

	pc_re = ProbNetSamp(sess, z, total, x_re, prob)
	return pc_re

#Message & Visualize
def WriteMessage(file, msg):
	print(msg)
	file.writelines(msg + "\n")

#========================= Latent Training Function =========================
def TrainLatent(grad, id_list, k_train, rate):
    for i in range(id_list.shape[0]):
        k_train[id_list[i]] -= rate * grad[i]

#========================= Data & Parameter =========================
sample_size = 20
latent_size = 8
repar_size = 8
batch_size = 300

k_train = np.random.normal(0., 0.001, [sample_size, latent_size])

#========================= Net Model =========================
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

#Placeholder
k_ = tf.placeholder(tf.float32, shape=[None, latent_size])
z_ = tf.placeholder(tf.float32, shape=[None, repar_size])
x_ = tf.placeholder(tf.float32, shape=[None, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

#Model Variable
W_mu = tf.Variable(xavier_init([latent_size, repar_size]))
b_mu = tf.Variable(tf.zeros(shape=[repar_size]))
W_sigma = tf.Variable(xavier_init([latent_size, repar_size]))
b_sigma = tf.Variable(tf.zeros(shape=[repar_size]))

W1 = tf.Variable(xavier_init([3+repar_size,512]))
b1 = tf.Variable(tf.zeros(shape=[512]))

W2 = tf.Variable(xavier_init([512,800]))
b2 = tf.Variable(tf.zeros(shape=[800]))

W3 = tf.Variable(xavier_init([800,1024]))
b3 = tf.Variable(tf.zeros(shape=[1024]))

W4 = tf.Variable(xavier_init([1024,512]))
b4 = tf.Variable(tf.zeros(shape=[512]))

W5 = tf.Variable(xavier_init([512,1]))
b5 = tf.Variable(tf.zeros(shape=[1]))

#Model Implement
def Reparameter(k):
	z_mu = tf.matmul(k, W_mu) + b_mu
	z_logvar = tf.matmul(k, W_sigma) + b_sigma
	return z_mu, z_logvar

def SampleZ(mu, log_var):
	eps = tf.random_normal(shape=tf.shape(mu))
	return mu + tf.exp(log_var / 2) * eps

def ProbNetGLO(x, z):
	x_z = tf.concat(axis=1, values=[x, z])
	h1 = tf.nn.relu(tf.matmul(x_z, W1) + b1)
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
	h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
	h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
	y_digit = tf.matmul(h4, W5) + b5
	y_prob = tf.nn.sigmoid(y_digit)
	return y_digit, y_prob

#Loss and optimizer
z_mu, z_logvar = Reparameter(k_)
z_sample = SampleZ(z_mu, z_logvar)
y_digit, y_prob = ProbNetGLO(x_, z_sample)
_, y_sample = ProbNetGLO(x_, z_)

recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_prob - y_), reduction_indices=[1]))
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
loss = tf.reduce_mean(recon_loss + kl_loss)

k_gradients = tf.gradients(loss, k_)
solver = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#========================= Main =========================
saveName = "chair_rglo_250.ckpt"
path = "3d_model/ModelNet10_chair/"
[pc_list, tree_list] = LoadDataFileId(path, sample_size)

# Record file
file = open("out/Record.txt", "a")
msg = "(" + saveName + ")"
WriteMessage(file, msg)

for i in range(400001):
	# Optimize
	kid = random.randint(0,sample_size-1)
	x_batch, y_batch = NextTrainBatch(sample_size, batch_size, pc_list[kid], tree_list[kid])
	k_batch = ZTensor(k_train[kid], batch_size)
	_, grad = sess.run([solver, k_gradients],  feed_dict={x_: x_batch, k_: k_batch, y_: y_batch})
	grad_np = np.asarray(grad[0])
	TrainLatent(grad_np, np.asarray([kid]), k_train, 1.)

	# Print message
	if i%10000 == 0:
		pc_re = ProbNetSampVoxel(sess, k_train[kid], 50, 0.94)
		msg = '[{}] Sample id {}, Size = {}'.format(str(datetime.now()), str(kid+1), str(len(pc_re)))
		WriteMessage(file, msg)
		if len(pc_re) < 15000:
			DrawPc(pc_re, show=False, filename="out/" + str(i))

	if i%1000 == 0:
		loss_ = sess.run(loss, feed_dict={x_: x_batch, k_: k_batch, y_: y_batch})
		msg = '{} Iter, Loss: {}'.format(str(i), loss_)
		WriteMessage(file, msg)

file.close()
print("Save parameter ...")
saver = tf.train.Saver()
save_path = saver.save(sess, "tf_save/" + saveName)