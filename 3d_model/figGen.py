from ReadPc import *
import os

path = "ModelNet10_chair/"
files = [f for f in os.listdir(path)]

for filename in files:
	f = filename.split(".")
	if len(f)==2 and f[1].strip()=="npts":
		pc = ReadNpts(path + filename)
		DrawPc(DownSample(pc, 10), show=False, filename="model_fig/" + f[0])
		print(f[0])
