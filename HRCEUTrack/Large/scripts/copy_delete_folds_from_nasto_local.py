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


data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/test"
save_path = r"/home/ioe/xxxxx/COESOT/test"
file_folds = os.listdir(save_path)

for videoID in range(len(file_folds)):
    foldname = file_folds[videoID]
    fileLIST = os.listdir(os.path.join(data_path, foldname))
    for filefold in fileLIST:
        src_fold = os.path.join(data_path, foldname, filefold)
        if os.path.isdir(src_fold):
            if 'timesurface' in filefold:
            # if 'recon' in filefold:
                print('copy the fold', foldname, filefold)
                new_fold = os.path.join(save_path, foldname, filefold)
                if not os.path.exists(new_fold):
                    os.makedirs(new_fold)
                    print('create fold', filefold)
                else:
                    continue
                for root, dirs, files in os.walk(src_fold):
                    for file in files:
                        src_file = os.path.join(root, file)
                        shutil.copy(src_file, new_fold)

                # remove
                # rm_path = os.path.join(save_path, foldname, filefold)
                # shutil.rmtree(rm_path)
