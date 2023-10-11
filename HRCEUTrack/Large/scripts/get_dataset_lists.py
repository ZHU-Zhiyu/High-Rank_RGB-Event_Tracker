import os
import shutil

data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/COESOT233/COESOT_external_sequences"
save_path = r"/home/ioe/xxxxx/EventTracking/COESOT/COESOT233/"
file_folds = os.listdir(data_path)

for videoID in range(len(file_folds)):
    foldname = file_folds[videoID]
    fileLIST = os.listdir(os.path.join(data_path, foldname))
    for file in fileLIST:
        if os.path.isdir(file):
            continue
        filename, extension = os.path.splitext(file)
        if (extension == '.aedat4'):
            with open(save_path + 'list233.txt', 'a+') as f:
                f.write(filename+'\n')

# train.txt  generate
# i = 0
# for i in range(0, 800):
#     with open(save_path + 'train.txt', 'a+') as f:
#         f.write(str(i)+'\n')
#         i=i+1
#
# # # val.txt generate
# for i in range(800, len(file_folds)):
#     with open(save_path + 'val.txt', 'a+') as f:
#         f.write(str(i)+'\n')
#         i=i+1


# data_path = r"/home/ioe/xxxxx/EventTracking/FE108/train"
# save_path = r"/home/ioe/xxxxx/EventTracking/FE108/"
# data_path = r"/home/ioe/xxxxx/EventTracking/Visevent/test"
# save_path = r"/home/ioe/xxxxx/EventTracking/Visevent/"
# file_folds = os.listdir(data_path)
#
# for videoID in range(len(file_folds)):
#     foldname = file_folds[videoID]
#     fileLIST = os.listdir(os.path.join(data_path, foldname))
#     for file in fileLIST:
#         if os.path.isdir(file):
#             continue
#         filename, extension = os.path.splitext(file)
#         if (extension == '.aedat4'):
#             mat_save = os.path.join(data_path, foldname, 'voxel/')
#             img_save = os.path.join(data_path, foldname, 'vis_imgs/')
#             if (len(os.listdir(mat_save)) == len(os.listdir(img_save))):
#                 with open(save_path + 'list.txt', 'a+') as f:
#                     f.write(filename+'\n')
