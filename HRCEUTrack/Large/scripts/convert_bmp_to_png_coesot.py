import shutil
import glob
from PIL import Image
import os
from pathlib import Path

data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train/"
save_path = data_path
match_file = data_path+'list.txt'
with open(match_file, 'r') as f:
    sequences_list = f.read()

video_files = os.listdir(data_path)
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    if foldName in sequences_list:
        print("==>> foldName: ", foldName)
        data_img_path = data_path+foldName+ '/'+foldName+ '_aps'
        save_img_path = data_img_path
        for img in glob.glob(data_img_path+"/*.bmp"):
            filename = Path(img).stem
            Image.open(img).save(save_img_path+f'/{filename}.png')
            os.remove(img)
            # print('convert img')
        # for img in glob.glob(data_img_path+"/*.bmp"):
            # if img.endswith(".bmp"):
                # print('removed bmps')
        # shutil.copytree(data_path+foldName+'/vis_imgs', )






