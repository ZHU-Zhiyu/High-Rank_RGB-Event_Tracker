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
from PIL import Image
import matplotlib


def to_timesurface_numpy(events, sensor_size=[346, 260, 3], surface_dimensions=None, tau=5e3, decay="lin"):
    # if surface_dimensions:
    #     assert len(surface_dimensions) == 2
    #     assert surface_dimensions[0] % 2 == 1 and surface_dimensions[1] % 2 == 1
    #     radius_x = surface_dimensions[0] // 2
    #     radius_y = surface_dimensions[1] // 2
    # else:
    radius_x = 0
    radius_y = 0
    surface_dimensions = sensor_size

    timestamp_memory = np.zeros((sensor_size[2], sensor_size[1] + radius_y * 2, sensor_size[0] + radius_x * 2))
    timestamp_memory -= tau * 3 + 1
    all_surfaces = np.zeros((len(events), sensor_size[2], surface_dimensions[1], surface_dimensions[0]))
    # time_surface_img = np.zeros((sensor_size[2], surface_dimensions[1], surface_dimensions[0]))
    for index, event in enumerate(events):
        x = int(event["x"])
        y = int(event["y"])
        timestamp_memory[int(event["polarity"]), y + radius_y, x + radius_x] = event["timestamp"]
        if radius_x > 0 and radius_y > 0:
            timestamp_context = (
                timestamp_memory[:, y : y + surface_dimensions[1], x : x + surface_dimensions[0]]- event["t"])
        else:
            timestamp_context = timestamp_memory - event["timestamp"]

        if decay == "lin":
            timesurface = timestamp_context / (3 * tau) + 1
            timesurface[timesurface < 0] = 0
        # elif decay == "exp":
        #     timesurface = np.exp(timestamp_context / tau)
        all_surfaces[index, :, :, :] = timesurface
        # time_surface_img += timesurface
    all_surfaces = np.sum(all_surfaces, axis=0)
    return all_surfaces
    # return time_surface_img


if __name__ == '__main__':
    device = torch.device("cuda:0")
    data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/test"
    save_path = data_path
    video_files = os.listdir(data_path)
    dvs_img_interval = 1

    for videoID in range(len(video_files)):
        foldName = video_files[videoID]
        print("==>> foldName: ", foldName)
        # if foldName in ['dvSave-2022_01_26_20_01_46', 'dvSave-2022_03_21_11_09_50',
        #                 'dvSave-2022_03_21_16_11_40', 'dvSave-2022_03_21_11_12_27',
        #                 'dvSave-2022_03_21_11_19_37', 'dvSave-2022_02_25_13_21_41',
        #                 'dvSave-2022_02_25_13_31_18', 'dvSave-2022_03_21_09_05_49']:
        #     print("==>> pass it: ", foldName)
        #     continue
        # if 'dvSave-2022_03_21' not in foldName:
        #     continue
        fileLIST = os.listdir(os.path.join(data_path, foldName))
        time_surface_path = os.path.join(save_path, foldName, foldName + '_timesurface')
        if not os.path.exists(time_surface_path):
            os.mkdir(time_surface_path)
        else:
            continue
        aedat4_file = foldName + '.aedat4'
        # print('filename:', aedat4_file)
        read_path = os.path.join(data_path, foldName, aedat4_file)

        # read aeda4;
        frame_all = []
        frame_exposure_time = []
        frame_interval_time = []
        with AedatFile(read_path) as f:
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_interval_time.append([frame.timestamp_start_of_frame,
                                            frame.timestamp_end_of_frame])  ## [1607928583387944, 1607928583410285]
        frame_timestamp = frame_interval_time
        frame_num = len(frame_timestamp)

        events = np.hstack([packet for packet in f['events'].numpy()])
        for frame_no in range(0, int(frame_num / dvs_img_interval) - 1):
            start_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][0])[0][0]
            end_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][1])[0][0]
            sub_event = events[start_idx:end_idx]
            time_surface_img = to_timesurface_numpy(sub_event)
            # cv2.imwrite('output.png', time_surface_img.transpose(1, 2, 0))
            # cv2.imshow('Event time surface', time_surface_img.transpose(1, 2, 0))
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(time_surface_path, 'frame{:04d}'.format(frame_no)+'.png'), time_surface_img.transpose(1, 2, 0))
