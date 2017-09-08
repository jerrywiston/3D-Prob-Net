import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
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
y = gaussian(x, 0.0, 0.03)
#y = uni(x, 0.0, 0.05)
plt.plot(x,y)
plt.show()
'''

dphi, dtheta = pi/250.0, pi/250.0
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
x = r*sin(phi)*cos(theta)
y = r*cos(phi)
z = r*sin(phi)*sin(theta)

s = mlab.mesh(x, y, z)
mlab.show()
