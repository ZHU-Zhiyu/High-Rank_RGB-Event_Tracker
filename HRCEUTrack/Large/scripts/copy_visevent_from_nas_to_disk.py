## copy the useful dataset information form NAS inot local disk.   Tang Chuanming 2022.09.12

from dv import AedatFile
import cv2
import os
import numpy as np
from PIL import Image
import pdb
import os
import shutil
import shutil

data_path = r"/home/ioe/xxxxx/EventTracking/Visevent/train"
save_path = r"/home/ioe/xxxxx/VisEvent/train"
list_fold = r"/home/ioe/xxxxx/EventTracking/Visevent/train/list.txt"
file_folds = os.listdir(data_path)

with open(list_fold, 'r') as f:
    video_names = f.readlines()
    for video in video_names:
        video = video.split('\n')[0]
        print('video:', video)
        new_voxel_fold = os.path.join(save_path, video, 'voxel')
        voxel_fold = os.path.join(data_path, video, 'voxel')
        if not os.path.exists(new_voxel_fold):
            os.makedirs(new_voxel_fold)
        for root, dirs, files in os.walk(voxel_fold):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, new_voxel_fold)

        new_vis_fold = os.path.join(save_path, video, 'vis_imgs')
        vis_fold = os.path.join(data_path, video, 'vis_imgs')
        if not os.path.exists(new_vis_fold):
            os.makedirs(new_vis_fold)
        for root, dirs, files in os.walk(vis_fold):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, new_vis_fold)


        new_gt_box = os.path.join(save_path, video, 'groundtruth.txt')
        gt_box = os.path.join(data_path, video, 'groundtruth.txt')
        shutil.copy(gt_box, new_gt_box)

# for videoID in range(len(file_folds)):
#     foldname = file_folds[videoID]
#     fileLIST = os.listdir(os.path.join(data_path, foldname))
#     for filefold in fileLIST:
#         src_fold = os.path.join(data_path, foldname, filefold)
#         if os.path.isdir(src_fold):
#             # if 'aps' in filefold or 'voxel' in filefold:
#             # if 'aps' in filefold or 'dvs' in filefold or 'voxel' in filefold:
#             if 'dvs' in filefold:
#                 print('copy the fold', filefold)
#                 new_fold = os.path.join(save_path, foldname, filefold)
#                 if not os.path.exists(new_fold):
#                     os.makedirs(new_fold)
#                     print('create fold', filefold)
#                 for root, dirs, files in os.walk(src_fold):
#                     for file in files:
#                         src_file = os.path.join(root, file)
#                         shutil.copy(src_file, new_fold)
#         elif os.path.splitext(filefold)[-1] == '.txt':
#             new_fold = os.path.join(save_path, foldname)
#             if not os.path.exists(new_fold):
#                 os.makedirs(new_fold)
#             shutil.copy(src_fold, os.path.join(new_fold, filefold))
