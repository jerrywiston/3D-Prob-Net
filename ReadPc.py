import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree

def ReadNpts(filename,num):
	file = open(filename, "r")
	content = file.read().strip().split("\n")
	
	total = len(content)
	if num>0 and num<total:
		total = num

	print("Total: " + str(total))
	pc = [0] * total

	for i in range(total):
		temp = content[i].split(" ")
		p1 = float(temp[0].strip())
		p2 = float(temp[1].strip())
		p3 = float(temp[2].strip())
		pc[i] = [p1, p2, p3]

	print("Read Npts Done !!")
	return pc


def DrawPc(pc, scale=[[-0.1, 0.1],[-0.1, 0.1],[-0.1, 0.1]]):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')	

	for p in pc:
		ax.scatter(p[0], p[1], p[2], c='b', marker='.')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	ax.set_aspect(1)
	ax.set_xlim(scale[0])
	ax.set_ylim(scale[1])
	ax.set_zlim(scale[2])
	plt.show()

def Rescale(pc):
	bound = [[999,-999],[999,-999],[999,-999]]
	for p in pc:
		for i in range(3):
			if p[i] < bound[i][0]:
				bound[i][0] = p[i]
			if p[i] > bound[i][1]:
				bound[i][1] = p[i]

	print("Bound: " + str(bound))
	
	dist = [0]*3
	for i in range(3):
		dist[i] = bound[i][1] - bound[i][0]

	rate = 0.0
	if dist[0] >= dist[1] and dist[0] >= dist[2]:
		rate = 1.0 / dist[0]
	elif dist[1] >= dist[0] and dist[1] >= dist[2]:
		rate = 1.0 / dist[1]
	else:
		rate = 1.0 / dist[2]
	
	for p in pc:
		for i in range(3):
			p[i] -= (bound[i][1] + bound[i][0])/2
			p[i] *= rate
			p[i] += 0.5

'''
pc = np.asarray(ReadNpts("horse.npts", 10000))
#pc = ReadNpts("horse.npts", 10000)
Rescale(pc)
DrawPc(pc,[[0,1],[0,1],[0,1]])

tree = KDTree(pc, leaf_size=2)
dist, ind = tree.query([pc[3]], k=3)
print(ind)
print(dist)
'''

