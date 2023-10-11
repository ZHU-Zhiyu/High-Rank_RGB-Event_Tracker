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
import cv2 as cv

# if __name__ == '__main__':
#     device = torch.device("cuda:4")
#     # data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
#     video_files = os.listdir(data_path)
#     dvs_img_interval = 1
#
#     for videoID in range(len(video_files)):
#         foldName = video_files[videoID]
#         print("==>> foldName: ", foldName)
#         aps_path = os.path.join(data_path, foldName, foldName + '_aps')
#         dvs_path = os.path.join(data_path, foldName, foldName + '_dvs')
#         stack_path = os.path.join(data_path, foldName, foldName + '_stack/')
#         # for i in tqdm(range(len(aps_path))):
#         frame_list = [frame for frame in os.listdir(aps_path)]
#         frame_list.sort(key=lambda f: int(f[-8:-4]))
#         frame_list_event = [frame for frame in os.listdir(dvs_path)]
#         frame_list_event.sort(key=lambda f: int(f[-8:-4]))
#         for frame, frame_event in zip(frame_list, frame_list_event):
#             image = cv.imread(os.path.join(aps_path, frame))
#             event_image = cv.imread(os.path.join(dvs_path, frame_event))
#             stack_image = cv.addWeighted(image, 1, event_image, 0.2, 0)
#             cv.imwrite(stack_path+frame, stack_image)
#


if __name__ == '__main__':
    device = torch.device("cuda:4")
    # data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
    data_path = r"/home/ioe/xxxxx/FE108/train"
    save_path = r"/home/ioe/xxxxx/FE108/train"
    video_files = os.listdir(data_path)
    dvs_img_interval = 1

    for videoID in range(len(video_files)):
        foldName = video_files[videoID]
        aps_path = os.path.join(data_path, foldName, 'aps')
        dvs_path = os.path.join(data_path, foldName, 'dvs')
        stack_path = os.path.join(data_path, foldName, 'stack/')
        if not os.path.exists(stack_path):
            os.mkdir(stack_path)
        else:
            continue
        print("==>> foldName: ", foldName)
        # for i in tqdm(range(len(aps_path))):
        frame_list = [frame for frame in os.listdir(aps_path)]
        frame_list.sort(key=lambda f: int(f[-8:-4]))
        frame_list_event = [frame for frame in os.listdir(dvs_path)]
        frame_list_event.sort(key=lambda f: int(f[-8:-4]))
        for frame, frame_event in zip(frame_list, frame_list_event):
            image = cv.imread(os.path.join(aps_path, frame))
            event_image = cv.imread(os.path.join(dvs_path, frame_event))
            stack_image = cv.addWeighted(image, 1, event_image, 0.1, 0)
            cv.imwrite(stack_path+frame, stack_image)
