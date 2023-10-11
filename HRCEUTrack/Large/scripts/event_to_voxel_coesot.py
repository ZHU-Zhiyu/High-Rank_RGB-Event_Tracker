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


def transform_points_to_voxels(data_dict={}, voxel_generator=None, device=torch.device("cuda:0")):
    """
    将点云转换为voxel,调用spconv的VoxelGeneratorV2
    """
    points = data_dict['points']
    # 将points打乱
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    data_dict['points'] = points

    # 使用spconv生成voxel输出
    points = torch.as_tensor(data_dict['points']).to(device)
    voxel_output = voxel_generator(points)

    # 假设一份点云数据是N*4，那么经过pillar生成后会得到三份数据
    # voxels代表了每个生成的voxel数据，维度是[M, 5, 4]
    # coordinates代表了每个生成的voxel所在的zyx轴坐标，维度是[M,3]
    # num_points代表了每个生成的voxel中有多少个有效的点维度是[m,]，因为不满5会被0填充
    voxels, coordinates, num_points = voxel_output
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # 选event数量在前5000的voxel  8000 from(4k+,6k+)
    # print(torch.where(num_points>=16)[0].shape)
    if num_points.shape[0] < save_voxel:
        features = voxels[:, :, 3]
        coor = coordinates[:, :]
    else:
        _, voxels_idx = torch.topk(num_points, save_voxel)
        # 将每个voxel的1024个p拼接作为voxel初始特征   16
        features = voxels[voxels_idx][:, :, 3]
        # 前5000个voxel的三维坐标
        coor = coordinates[voxels_idx]
    # 将y.x.t改为t,x,y
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]

    return coor, features


if __name__ == '__main__':
    save_voxel = 5000
    use_mode = 'frame_exposure_time'
    device = torch.device("cuda:0")
    # data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
    data_path = r"/data0/zhu_19/COETracking/train_subset"
    save_path = r"/data1/zhu_19/COESOT/trainning_voxel"
    # save_path = r"/home/ioe/xxxxx/COESOT/train"
    video_files = os.listdir(data_path)
    dvs_img_interval = 1
    voxel_generator = PointToVoxel(
        # 给定每个voxel的长宽高  [0.05, 0.05, 0.1]
        vsize_xyz=[50, 10, 10],  # [0.2, 0.25, 0.16]  # [50, 10, 10]  [50, 35, 26] 因此坐标范围（20,20,20）  (20, 34/35, 26)
        # 给定点云的范围 [  0.  -40.   -3.   70.4  40.    1. ]
        coors_range_xyz=[0, 0, 0, 1000, 345, 259],
        # 给定每个点云的特征维度，这里是x，y，z，r 其中r是激光雷达反射强度       # 346x260  t,x,y
        num_point_features=4,
        # 最多选取多少个voxel，训练16000，推理40000
        max_num_voxels=16000,  # 16000
        # 给定每个pillar中有采样多少个点，不够则补0  因此我将neg voxel改为-1;
        max_num_points_per_voxel=16,  # 1024
        device=device
    )

    for videoID in range(len(video_files)):
        foldName = video_files[videoID]
        print("==>> foldName: ", foldName)
        if foldName in ['dvSave-2022_01_26_20_01_46', 'dvSave-2022_03_21_11_09_50',
                        'dvSave-2022_03_21_16_11_40', 'dvSave-2022_03_21_11_12_27',
                        'dvSave-2022_03_21_11_19_37', 'dvSave-2022_02_25_13_21_41',
                        'dvSave-2022_02_25_13_31_18', 'dvSave-2022_03_21_09_05_49']:
            print("==>> pass it: ", foldName)
            continue
        # if 'dvSave-2022_03_21' not in foldName:
        #     continue
        fileLIST = os.listdir(os.path.join(data_path, foldName))
        if not os.path.exists(os.path.join(save_path, foldName, foldName+'_voxel')):
            os.mkdir(os.path.join(save_path, foldName, foldName+'_voxel'))
        else:
            continue
        mat_save = os.path.join(save_path, foldName, foldName+'_voxel/')

        # for FileID in tqdm(range(len(fileLIST))):
        #     videoName = fileLIST[FileID]
        #     (filename, extension) = os.path.splitext(videoName)
        #     if (extension == '.aedat4') :  #  and ('dvSave-2022_03_21' in foldName):
        #         # if os.path.exists(mat_save + '{}.mat'.format(filename)):
        #         #     print("==>> Skip this video ... ")
        #         #     continue
        aedat4_file = foldName + '.aedat4'
        # print('filename:', aedat4_file)
        read_path = os.path.join(data_path, foldName, aedat4_file)

        # read aeda4;
        frame_all = []
        frame_exposure_time = []
        frame_interval_time = []
        with AedatFile(read_path) as f:
            # print(f.names)
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_exposure_time.append([frame.timestamp_start_of_exposure,
                                            frame.timestamp_end_of_exposure])  ## [1607928583397102, 1607928583401102]
                frame_interval_time.append([frame.timestamp_start_of_frame,
                                            frame.timestamp_end_of_frame])  ## [1607928583387944, 1607928583410285]
        if use_mode == 'frame_exposure_time':
            frame_timestamp = frame_exposure_time
        elif use_mode == 'frame_interval_time':
            frame_timestamp = frame_interval_time
        frame_num = len(frame_timestamp)

        events = np.hstack([packet for packet in f['events'].numpy()])

        t_all = torch.tensor(events['timestamp']).unsqueeze(1).to(device)
        x_all = torch.tensor(events['x']).unsqueeze(1).to(device)
        y_all = torch.tensor(events['y']).unsqueeze(1).to(device)
        p_all = torch.tensor(events['polarity']).unsqueeze(1).to(device)

        for frame_no in range(0, int(frame_num / dvs_img_interval) - 1):
            start_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][0])[0][0]
            end_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][1])[0][0]
            sub_event = events[start_idx:end_idx]
            idx_length = end_idx - start_idx

            t = t_all[start_idx: end_idx]
            if start_idx == end_idx:
                time_length = 0
                t = ((t-t).float() / time_length) * 1000
                scipy.io.savemat(mat_save + 'frame{:0>4d}.mat'.format(frame_no), mdict={'coor': np.zeros([100, 3]),
                                 'features': np.zeros([100, 16])})  # coor: numpy.nan;   features: numpy.nan
                print('empty event frame ', frame_no)
                continue
            else:
                time_length = t[-1] - t[0]
                # rescale the timestampes to start from 0 up to 1000
                t = ((t-t[0]).float() / time_length) * 1000
            all_idx = np.where(sub_event['polarity'] != -1)      # all event
            neg_idx = np.where(sub_event['polarity'] == 0)      # neg event
            t = t[all_idx]
            x = x_all[all_idx]
            y = y_all[all_idx]
            p = p_all[all_idx]
            p[neg_idx] = -1     # negtive voxel change from 0 to -1. because after append 0 operation.
            current_events = torch.cat((t, x, y, p), dim=1)
            # if current_events.shape[0] < 10:   # remove it
            #     continue
            data_dict = {'points': current_events}

            coor, features = transform_points_to_voxels(data_dict=data_dict, voxel_generator=voxel_generator,
                                                        device=device)
            # pdb.set_trace()
            coor = coor.cpu().numpy()
            features = features.cpu().numpy()
            # print('coor', coor)
            scipy.io.savemat(mat_save + 'frame{:0>4d}.mat'.format(frame_no), mdict={'coor': coor, 'features': features})   # coor: Nx(t,x,y);   features:Nx32 or Nx10024
