import os
import cv2
import torch
from dv import AedatFile
import numpy as np

def to_timesurface_numpy(events, sensor_size=[346, 260, 3], tau=10, decay="lin"):
    timestamp_memory = np.zeros((sensor_size[2], sensor_size[1], sensor_size[0]))
    timestamp_memory -= tau * 3 + 1
    x = events["x"]
    y = events["y"]
    timestamp_memory[events["polarity"], y, x] = events["timestamp"]
    timestamp_context = timestamp_memory - events["timestamp"].mean()
    timesurface = timestamp_context / (3 * tau) + 1
    timesurface[timesurface < 0] = 0
    all_surfaces = timesurface
    return all_surfaces



if __name__ == '__main__':
    device = torch.device("cuda:0")
    data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
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
        fileLIST = os.listdir(os.path.join(data_path, foldName))
        time_surface_path = os.path.join(save_path, foldName, foldName + '_timesurface')
        if not os.path.exists(time_surface_path):
            os.mkdir(time_surface_path)
        else:
            continue
        aedat4_file = foldName + '.aedat4'
        read_path = os.path.join(data_path, foldName, aedat4_file)

        # read aeda4;
        frame_all = []
        frame_exposure_time = []
        frame_interval_time = []
        use_mode = 'frame_exposure_time'
        with AedatFile(read_path) as f:
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_interval_time.append([frame.timestamp_start_of_frame, frame.timestamp_end_of_frame])
                frame_exposure_time.append([frame.timestamp_start_of_exposure,
                                            frame.timestamp_end_of_exposure])
        if use_mode == 'frame_exposure_time':
            frame_timestamp = frame_exposure_time
        elif use_mode == 'frame_interval_time':
            frame_timestamp = frame_interval_time
        frame_num = len(frame_timestamp)

        events = np.hstack([packet for packet in f['events'].numpy()])
        for frame_no in range(0, int(frame_num / dvs_img_interval) - 1):
            start_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][0])[0][0]
            end_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][1])[0][0]
            time_surface_img = to_timesurface_numpy(events[start_idx:end_idx])
            cv2.imwrite(os.path.join(time_surface_path, 'frame{:04d}'.format(frame_no)+'.png'),
                        time_surface_img.transpose(1, 2, 0))
