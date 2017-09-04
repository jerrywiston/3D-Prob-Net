import os

files = [f for f in os.listdir(".")]
pc_list = []
tree_list = []

i = 0
for filename in files:
    f = filename.split(".")
    if len(f)==2 and f[1].strip()=="png":
    	print(i)
    	os.rename(filename, str(i)+"_zsamp.png")
    	i+=1