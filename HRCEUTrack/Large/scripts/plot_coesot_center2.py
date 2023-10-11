import shutil
import glob

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import  *
import pandas as pd

data_path = r"/home/ioe/xxxxx/COESOT/test/"
match_file = data_path+'list.txt'
with open(match_file, 'r') as f:
    sequences_list = f.read()

video_files = os.listdir(data_path)
plot_list = np.zeros([346, 260])
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    if foldName in sequences_list:
        print("==>> foldName: ", foldName)
        txt_path = data_path + foldName + '/groundtruth.txt'
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                center = line[:-1].split(',')
                x, y = int(float(center[0])), int(float(center[1]))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if x > 345:
                    x = 345
                if y > 259:
                    y = 259
                plot_list[x][y] += 1
                # print(center)

from mpl_toolkits.mplot3d import axes3d
plot_list[0:10,] = 0
plot_list[:,0:10] = 0
plot_list[np.where(plot_list > 5)] = 5
plot_list = gaussian_filter(plot_list, sigma=3)
# plot_list[np.where(plot_list > 50 & (plot_list < 1000))] = 50

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.mgrid[0:346, 0:260]
Z = plot_list
ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, alpha=0.5)
ax.contour(X, Y, Z, 10, cmap="autumn_r", linestyles='solid')
# ax.contour(X, Y, Z, 10, colors="k", linestyles='solid')
plt.show()





