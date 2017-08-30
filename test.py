import tensorflow as tf
import numpy as np

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