import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import pandas as pd

W = 720
H = 576 
g_size = 36

grid_W = W//g_size
grid_H = H//g_size

obj_info = np.zeros([grid_H,grid_W])
for i in range(round(249/g_size)):
    for j in range(round(442/g_size)):
        obj_info[i,j]=1

for i in range(round(300/g_size)):
    for j in range(round(467/g_size), round(562/g_size)):
        obj_info[i,j]=1

for i in range(round(490/g_size),grid_H):
    for j in range(grid_W):
        obj_info[i,j]=1

#df = pd.DataFrame(data=obj_info.astype(int))
#df.to_csv('Code/pix2met/obj_info.csv', sep=' ', header=False, float_format='%.2f', index=False)

scene = mimg.imread('Code/vgg/frame_1.png')
im1 = plt.imshow(scene, extent = [0,576,0,720])

im2 = plt.imshow(obj_info, alpha=0.5, extent = [0,576,0,720])
plt.show()
