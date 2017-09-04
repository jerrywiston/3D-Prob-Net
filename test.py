import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
'''
x_ = tf.placeholder(tf.float32, shape=[3])
y = tf.nn.l2_normalize(x_, dim=0, epsilon=1)

sess = tf.Session()
x = [0]*3
x[0] = 1
x[1] = 2
x[2] = 3
yy = sess.run(y, feed_dict={x_: np.asarray(x)})
print(yy)
print(yy.dot(yy))

'''
def gaussian(x, u, sig):
	y = np.exp(-(x - u) ** 2 /(2* sig **2))#/(math.sqrt(2*math.pi)*sig)
	return y 

def uni(x, u, r):
	d = abs(x-u)
	y = [0.0]*d.shape[0]
	for i in range(d.shape[0]):
		if d[i] < r:
			y[i] = (r - abs(x[i]-u))/r
	return y

x = np.linspace(-0.5, 0.5, num=1000)
#y = gaussian(x, 0.0, 0.05)
y = uni(x, 0.0, 0.05)
print(np.exp(0))
#print(gaussian(1.3, 0, 1.0))
plt.plot(x,y)
plt.show()

