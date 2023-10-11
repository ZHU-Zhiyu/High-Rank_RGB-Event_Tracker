import os
import pdb
# import csv
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io
from spconv.pytorch.utils import PointToVoxel
from dv import AedatFile
import numpy as np

# data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
data_path = r"/media/ioe/2t/COESOT_external_sequences"

video_files = os.listdir(data_path)
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    # print("==>> finished: ", foldName)
    read_path = os.path.join(data_path, foldName, 'groundtruth.txt')
    f = open(read_path)
    number_0 = 0
    number_no0 = 0
    for line in f:
        H, W = line[:-1].split(',')[2], line[:-1].split(',')[3]
        if float(H) * float(W) < 1:
            number_0 += 1
        else:
            number_no0 += 1
    if number_0 > 10 and (number_0 > number_no0):
        print("==>> too many empty : ", foldName, number_0, number_no0)
