import shutil
import glob
from PIL import Image
import os
from pathlib import Path

data_path = r"/home/ioe/xxxxx/EventTracking/Visevent/train/"
save_path = r"/home/ioe/xxxxx/VisEvent/train/"
match_file = '/home/ioe/xxxxx/EventTracking/Visevent/train/list.txt'
with open(match_file, 'r') as f:
    sequences_list = f.read()

video_files = os.listdir(data_path)
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    if foldName in sequences_list:
        print("==>> foldName: ", foldName)
        if not os.path.exists(save_path+foldName):
            os.mkdir(save_path+foldName)
        if not os.path.exists(save_path+foldName+'/voxel'):
            shutil.copytree(data_path+foldName+'/voxel', save_path+foldName+'/voxel')
        if not os.path.exists(save_path+foldName+'/groundtruth.txt'):
            shutil.copy(data_path+foldName+'/groundtruth.txt', save_path+foldName+'/groundtruth.txt')

        data_img_path = data_path+foldName+'/vis_imgs'
        save_img_path = save_path+foldName+'/vis_imgs'
        if not os.path.exists(save_img_path):
            os.mkdir(save_img_path)
        else:
            continue
        for img in glob.glob(data_img_path+"/*.bmp"):
            filename = Path(img).stem
            Image.open(img).save(save_img_path+f'/{filename}.png')


        # shutil.copytree(data_path+foldName+'/vis_imgs', )






