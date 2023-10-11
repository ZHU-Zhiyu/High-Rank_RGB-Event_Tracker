import os
import pdb

data_path = r"/home/ioe/xxxxx/COESOT/test"
frame_num=0
video_files = os.listdir(data_path)
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    # print("==>> finished: ", foldName)
    read_path = os.path.join(data_path, foldName, foldName+'_dvs')
    lenth = len(os.listdir(read_path))
    frame_num += lenth
print(frame_num)

#  train: 290514     test: 188207   total: 478721
