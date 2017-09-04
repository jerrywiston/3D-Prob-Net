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

'''
### Two key samp test 
k = [0.0]*totalModel
k[8] = 0.5
k[13] = 0.5
pc_re = ProbNetSamp(key=k, samp=100000, prob=0.92)
print(len(pc_re))
DrawPc(pc_re,[[0,1],[0,1],[0,1]])


### Reconstruct test
for i in range(100):
	pc_re = ProbNetIdSamp(kid=i, samp=100000, prob=0.94)
	print(len(pc_re))
	DrawPc(pc_re,[[0,1],[0,1],[0,1]],  show=False, filename="out/reconst/" + str(i+1) + "_reconst")

### k samp test
for i in range(100):
	k = OneKeyRand(totalModel)
	pc_re = ProbNetSamp(key=k, samp=100000, prob=0.94)
	print(k)

	print(len(pc_re))
	DrawPc(pc_re,[[0,1],[0,1],[0,1]])
'''
### z samp test
for i in range(25,100):
	z = OneZRand(z_dim)
	#pc_samp = ZSamp(z, 100000, 0.9)
	pc_samp = ZSampVoxel(z, 50, 0.95)
	print("Total Point: " + str(len(pc_samp)))
	#DrawPc(pc_samp,[[0,1],[0,1],[0,1]])
	DrawPc(pc_samp,[[0,1],[0,1],[0,1]], show=False, filename="out/samp/" + str(i) + "_samp")

'''
# Interpolation test
#38,48
#46,83
#55,58
k1 = [0.0]*totalModel
k1[55] = 1.0
k2 = [0.0]*totalModel
k2[58] = 1.0
z_e = sess.run(z_encode, feed_dict={k_:np.asarray([k1, k2])})

for i in range(6):
	z = 0.2*i*z_e[1] + (1-0.2*i)*z_e[0]
	z_samp = Znormalize(z.tolist())
	pc_samp = ZSamp(z_samp, 100000, 0.9)
	print("Total Point: " + str(len(pc_samp)))
	DrawPc(pc_samp,[[0,1],[0,1],[0,1]], show=False, filename="out/interpo/" + str(i) + "_interpo")



for i in range(100):
	pc_re = ProbNetIdSampVoxel(kid=i, vox_len=50, prob=0.92)
	print(len(pc_re))
	DrawPc(pc_re,[[0,1],[0,1],[0,1]])

'''