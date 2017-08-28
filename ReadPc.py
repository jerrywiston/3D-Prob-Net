import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.neighbors import KDTree

def DrawPc(pc, scale=[[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]):
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

# Npts
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

def TrainSampNpts(filename):
	pc = ReadNpts(filename, -1)
	Rescale(pc)
	return pc

# CAD
def SurfaceArea(p1, p2, p3):
	vec1 = np.asarray([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
	vec2 = np.asarray([p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]])
	cross_vec = np.cross(vec1, vec2)
	area = np.sqrt(cross_vec.dot(cross_vec))
	return area/2

def SurfaceSamp(p1, p2, p3, total):
	pc = [p1, p2, p3]
	if total <=3:
		return pc

	for i in range(total-3):
		r = np.random.uniform(0., 1., size=[1])[0]
		w1 = np.random.uniform(0., r, size=[1])[0]
		w2 = r - w1

		vec1 = np.asarray([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
		vec2 = np.asarray([p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]])

		p = np.asarray(p1) + w1*vec1 + w2*vec2
		pc.append([p[0], p[1], p[2]])

	return pc	

def SampFromCAD(v, s):
	magic_rate = 0.00005
	pc = []
	for i in range(len(s)):
		p1 = v[s[i][1]]
		p2 = v[s[i][2]]
		p3 = v[s[i][3]]
		sArea = SurfaceArea(p1, p2, p3)
		pc.extend(SurfaceSamp(p1, p2, p3, int(sArea/magic_rate)))
	return pc

def ReadCAD(filename):
	file = open(filename, "r")
	content = file.read().strip().split("\n")
	v_total = int(content[1].split(" ")[0])
	s_total = int(content[1].split(" ")[1])
	print(content[0])
	print(v_total)
	print(s_total)

	vertics = [0]*v_total
	for i in range(v_total):
		line = content[i+2].strip().split(" ")
		vertics[i] = [float(line[0]), float(line[1]), float(line[2])]

	surfaces = [0]*s_total
	for i in range(s_total):
		line = content[i+2+v_total].strip().split(" ")
		surfaces[i] = [int(line[0])]

		for j in range(surfaces[i][0]):
			surfaces[i].append(int(line[j+1]))

	return vertics, surfaces

def TrainSampCAD(filename):
	v, s = ReadCAD(filename)
	Rescale(v)
	pc = SampFromCAD(v, s)
	return pc

'''
pc = np.asarray(ReadNpts("horse.npts", 10000))
#pc = ReadNpts("horse.npts", 10000)
Rescale(pc)
DrawPc(pc,[[0,1],[0,1],[0,1]])

tree = KDTree(pc, leaf_size=2)
dist, ind = tree.query([pc[3]], k=3)
print(ind)
print(dist)

p1 = [0.0,0.0,0.0]
p2 = [1.0,0.0,0.0]
p3 = [0.0,1.0,0.0]
pc = SurfaceSamp(p1,p2,p3,1000)
DrawPc(pc)

pc = TrainSampCAD("chair_0001.off")
print(len(pc))
DrawPc(pc)
'''