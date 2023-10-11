import os
import shutil

data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/video_seq_test"
save_path = r"/home/ioe/xxxxx/EventTracking/COESOT/videos/video_seq_test/"
file_folds = os.listdir(data_path)


for videoID in range(len(file_folds)):
    foldname = file_folds[videoID]
    fileLIST = os.listdir(os.path.join(data_path, foldname))
    for file in fileLIST:
        # if os.path.isdir(file):
        #     continue
        filename, extension = os.path.splitext(file)
        if (extension == '.avi'):
            shutil.copy(os.path.join(data_path, foldname, file), os.path.join(save_path, file))
            with open(save_path + 'test_videos.txt', 'a+') as f:
                f.write(filename+'\n')



