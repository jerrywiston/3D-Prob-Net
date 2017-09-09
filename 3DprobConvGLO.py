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
	y_re = sess.run(y_prob, feed_dict={x_: x_re, z_: z_re})
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
def LatentRescale(z):
    length = np.sqrt(z.dot(z))
    if length > 1.0:
        z /= length
    return z

def TrainLatent(grad, id_list, z_train, rate):
    for i in range(id_list.shape[0]):
        z_update = z_train[id_list[i]] - rate * grad[i]
        z_train[id_list[i]] = LatentRescale(z_update)

#========================= Data & Parameter =========================
sample_size = 100
latent_size = 10
batch_size = 300

z_train = np.random.normal(0., 0.5, [sample_size, latent_size])

#========================= Net Model =========================
def xavier_init(size):
	if len(size) == 4:
		n_inputs = size[0]*size[1]*size[2]
		n_outputs = size[3]
	else:
		n_inputs = size[0]
		n_outputs = size[1]
	
	stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
	return tf.truncated_normal(size, stddev=stddev)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

#Placeholder
z_ = tf.placeholder(tf.float32, shape=[None, latent_size])
x_ = tf.placeholder(tf.float32, shape=[None, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

#Model Variable
W_fc1 = tf.Variable(xavier_init([3+latent_size,1024]))
b_fc1 = tf.Variable(tf.zeros(shape=[1024]))

W_conv2 = tf.Variable(xavier_init([3,3,1,16]))
b_conv2 = tf.Variable(tf.zeros(shape=[16]))

W_conv3 = tf.Variable(xavier_init([3,3,16,16]))
b_conv3 = tf.Variable(tf.zeros(shape=[16]))

W_fc4 = tf.Variable(xavier_init([8*8*16,512]))
b_fc4 = tf.Variable(tf.zeros(shape=[512]))

W_fc5 = tf.Variable(xavier_init([512,1]))
b_fc5 = tf.Variable(tf.zeros(shape=[1]))

#Model Implement
def ProbNetGLO(x, z):
	x_z = tf.concat(axis=1, values=[x, z])

	h_fc1 = tf.nn.relu(tf.matmul(x_z, W_fc1) + b_fc1)
	h_re1 = tf.reshape(h_fc1, [-1,32,32,1])

	h_conv2 = tf.nn.relu(conv2d(h_re1, W_conv2) + b_conv2)
	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
	h_re3 = tf.reshape(h_conv3, [-1,8*8*16])

	h_fc4 = tf.nn.relu(tf.matmul(h_re3, W_fc4) + b_fc4)

	y_digit = tf.matmul(h_fc4, W_fc5) + b_fc5
	y_prob = tf.nn.sigmoid(y_digit)
	return y_digit, y_prob

#Loss and optimizer
y_digit, y_prob = ProbNetGLO(x_, z_)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_prob - y_), reduction_indices=[1]))
z_gradients = tf.gradients(loss, z_)
solver = tf.train.AdamOptimizer().minimize(loss)

#Build model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#========================= Main =========================
saveName = "chair_glo_250.ckpt"
path = "3d_model/ModelNet10_chair/"
[pc_list, tree_list] = LoadDataFileId(path, sample_size)

# Record file
file = open("out/Record.txt", "a")
msg = "(" + saveName + ")"
WriteMessage(file, msg)

for i in range(400001):
	# Optimize
	zid = random.randint(0,sample_size-1)
	x_batch, y_batch = NextTrainBatch(sample_size, batch_size, pc_list[zid], tree_list[zid])
	z_batch = ZTensor(z_train[zid], batch_size)
	_, grad = sess.run([solver, z_gradients],  feed_dict={x_: x_batch, z_: z_batch, y_: y_batch})
	grad_np = np.asarray(grad[0])
	TrainLatent(grad_np, np.asarray([zid]), z_train, 1.)

	# Print message
	if i%1000 == 0:
		pc_re = ProbNetSampVoxel(sess, z_train[zid], 28, 0.94)
		pc_samp = ProbNetSampVoxel(sess, ZSamp(latent_size), 28, 0.94)
		msg = '[{}] Sample id {}, Size = {}'.format(str(datetime.now()), str(zid+1), str(len(pc_re)))
		WriteMessage(file, msg)
		DrawPc(pc_re, show=False, filename="out/{}_recon".format(str(i)))
		DrawPc(pc_samp, show=False, filename="out/{}_sample".format(str(i)))

	if i%100 == 0:
		loss_ = sess.run(loss, feed_dict={x_: x_batch, z_: z_batch, y_: y_batch})
		msg = '{} Iter, Loss: {}'.format(str(i), loss_)
		WriteMessage(file, msg)

file.close()
print("Save parameter ...")
saver = tf.train.Saver()
save_path = saver.save(sess, "tf_save/" + saveName)