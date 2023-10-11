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
plot_list[0:7,] = 0
plot_list[:,0:7] = 0
# plot_list[np.where(plot_list > 50)] = 51
plot_list[np.where(plot_list > 7)] = 7

# plot_list = gaussian_laplace(plot_list, sigma=3)
plot_list = gaussian_filter(plot_list, sigma=3)

sns.set()
import palettable
ax = sns.heatmap(plot_list, cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors, center=4, cbar=True, vmin=np.min(plot_list), vmax=np.max(plot_list), cbar_kws={'ticks':[0,25,50,75,125]})    # coolwarm
# plt.pie(x, colors = sns.color_palette('pastel'))
plt.axis('off')
plt.show()





