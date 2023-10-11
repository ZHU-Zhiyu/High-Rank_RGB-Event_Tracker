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


if __name__ == '__main__':
    device = torch.device("cuda:4")
    data_path = r"/home/ioe/xxxxx/EventTracking/Visevent/train"
    save_path = r"/home/ioe/xxxxx/EventTracking/Visevent/train"
    video_files = os.listdir(data_path)

    for videoID in range(len(video_files)):
        foldName = video_files[videoID]
        print("==>> foldName: ", foldName)
        fileLIST = os.listdir(os.path.join(data_path, foldName))
        if not os.path.exists(os.path.join(save_path, foldName, 'voxel')):
            os.mkdir(os.path.join(save_path, foldName, 'voxel'))
        mat_save = os.path.join(save_path, foldName, 'voxel/')
        img_save = os.path.join(save_path, foldName, 'vis_imgs/')
        if os.path.exists(mat_save) and (len(os.listdir(mat_save)) > len(os.listdir(img_save))):
            print('voxel too much or ')
            continue
        elif os.path.exists(mat_save) and (len(os.listdir(mat_save)) == 0):
            print('voxel  is zero. ')
            continue
        elif os.path.exists(mat_save) and (len(os.listdir(mat_save)) == len(os.listdir(img_save))):
            print('right voxel number.')
            with open(save_path + 'list.txt', 'a+') as f:
                f.write(foldName+'\n')
