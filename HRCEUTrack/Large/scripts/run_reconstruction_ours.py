# -*-coding:utf-8-*-
import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
import os
import cv2
import pdb
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
from PIL import Image
from dv import AedatFile


def extract_davis(aedat_file_path, dvs_img_interval):
    frame_no = 0
    frame_all = []
    frame_exposure_time = []
    frame_interval_time = []
    # use_mode = 'frame_exposure_time'  #### default mode
    use_mode = 'frame_interval_time'
    with AedatFile(aedat_file_path) as f:
        # list all the names of streams in the file
        print(f.names)
        # extract timestamps of each frame
        for frame in f['frames']:
            frame_all.append(frame.image)
            frame_exposure_time.append([frame.timestamp_start_of_exposure,
                                        frame.timestamp_end_of_exposure])  ## [1607928583397102, 1607928583401102]
            frame_interval_time.append(
                [frame.timestamp_start_of_frame, frame.timestamp_end_of_frame])  ## [1607928583387944, 1607928583410285]

            # pdb.set_trace()

        if use_mode == 'frame_exposure_time':
            frame_timestamp = frame_exposure_time
        elif use_mode == 'frame_interval_time':
            frame_timestamp = frame_interval_time

        frame_num = len(frame_timestamp)
        # Access dimensions of the event stream
        height, width = f['events'].size  # 260  346
        event_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
        # loop through the "events" stream
        # events = np.hstack([packet for packet in f['events'].numpy()])

        # # Access information of all events by type
        # timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        # pdb.set_trace()
        # save event_img
        idx = np.round(np.linspace(0, len(frame_timestamp) - 1, int(frame_num / dvs_img_interval))).astype(
            int)  ## frame index [0, 1, 2, ... , 3847]
        frame_timestamp = np.array(frame_timestamp)[idx]  ## (3848, 2)

        # another reader # speed up
        # events will be a named numpy array
        events = np.hstack([packet for packet in f['events'].numpy()])
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']

        return timestamps, x, y, polarities, frame_num, frame_timestamp

    #######################################################################################################


####                                Main Function
#######################################################################################################
if __name__ == "__main__":
    device = torch.device("cuda:4")
    # data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
    data_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
    save_path = r"/home/ioe/xxxxx/EventTracking/COESOT/train"
    video_files = os.listdir(data_path)
    for videoID in range(len(video_files)):
        foldName = video_files[videoID]
        print("==>> foldName: ", foldName)
        if foldName in ['dvSave-2022_01_26_20_01_46', 'dvSave-2022_03_21_11_09_50',
                        'dvSave-2022_03_21_16_11_40', 'dvSave-2022_03_21_11_12_27',
                        'dvSave-2022_03_21_11_19_37', 'dvSave-2022_02_25_13_21_41',
                        'dvSave-2022_02_25_13_31_18', 'dvSave-2022_03_21_09_05_49']:
            print("==>> pass it: ", foldName)
            continue

        aedat_file_path = os.path.join(data_path, foldName, foldName+'.aedat4')
        save_img_path = os.path.join(save_path, foldName, foldName+'_recon')
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
    # dvs_img_interval = 1
    # for root, dirs, files in os.walk(aedat_path):
    #     for file in files:
    #         (filename, extension) = os.path.splitext(file)
            # print("==>> filename: ", filename)
            # if (extension == '.aedat4'):
            #     if filename + '_aps' in os.listdir(root) or filename + '_dvs' in os.listdir(root):
            #         print("==>> Skip this video ... ")
            #         continue

                    # save_path_dvs = os.path.join(root, filename, filename+'_dvs')
                # save_path_aps = os.path.join(root, filename, filename+'_aps')
                # aedat_file_path = os.path.join(root, filename + '.aedat4')
                # filename_txt_path = os.path.join(root, filename+'_timestamp.txt')
                # croped_events_path = os.path.join(root, filename+'_events.txt')

                # if not os.path.exists(save_path_dvs):
                #     os.makedirs(save_path_dvs)
                # if not os.path.exists(save_path_aps):
                #     os.makedirs(save_path_aps)

        timestamps, x, y, polarities, frame_num, frame_timestamp = extract_davis(aedat_file_path, 1)  ## (4391964,)

        # all_events = [timestamps, x, y, polarities]

        all_events = np.zeros((timestamps.shape[0], 4))

        all_events[:, 0] = timestamps
        all_events[:, 1] = x
        all_events[:, 2] = y
        all_events[:, 3] = polarities
        # pdb.set_trace()

        parser = argparse.ArgumentParser(description='Evaluating a trained network')
        parser.add_argument('-c', '--path_to_model', required=True, type=str, help='path to model weights')
        parser.add_argument('-i', '--input_file', required=True, type=str)
        parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
        parser.set_defaults(fixed_duration=False)
        parser.add_argument('-N', '--window_size', default=500000, type=int,
                            help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
        parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                            help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
        parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                            help='in case N (window size) is not specified, it will be automatically computed as N = width * height * num_events_per_pixel')
        parser.add_argument('--skipevents', default=0, type=int)
        parser.add_argument('--suboffset', default=0, type=int)
        parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu',
                            action='store_true')
        parser.set_defaults(compute_voxel_grid_on_cpu=False)

        set_inference_options(parser)

        args = parser.parse_args()

        # Load model
        device = get_device(args.use_gpu)
        model = load_model(args.path_to_model, device)

        model = model.to(device)
        model.eval()

        # pdb.set_trace()
        height, width = 260, 346
        reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)
        start_s = 0
        """ Read chunks of events using Pandas """
        for frame_no in range(0, frame_num-1):
            # Loop through the events and reconstruct images
            N = frame_timestamp[frame_no][1] - frame_timestamp[frame_no][0]

            # if not args.fixed_duration:
            #     if N is None:
            #         N = int(width * height * args.num_events_per_pixel)
            #         print(
            #             'Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
            #                 N, args.num_events_per_pixel))
            #     else:
            #         print('Will use {} events per tensor (user-specified)'.format(N))
            #         mean_num_events_per_pixel = float(N) / float(width * height)
            #         if mean_num_events_per_pixel < 0.1:
            #             print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
            #                 The reconstruction results might be suboptimal.'.format(N))
            #         elif mean_num_events_per_pixel > 1.5:
            #             print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
            #                 The reconstruction results might be suboptimal.'.format(N))

            initial_offset = args.skipevents
            sub_offset = args.suboffset
            start_index = initial_offset + sub_offset

            if args.compute_voxel_grid_on_cpu:
                print('Will compute voxel grid on CPU.')

            with Timer('Processing entire dataset'):
                # event_window = all_events[:N, :]
                # event_window = all_events[start_s: start_s+N, :]
                # last_timestamp = event_window[-1, 0]
                # start_s += N

                start_idx = np.where(all_events[:, 0] >= frame_timestamp[frame_no][0])[0][0]
                end_idx = np.where(all_events[:, 0] >= frame_timestamp[frame_no][1])[0][0]
                event_window = all_events[start_idx:end_idx]
                last_timestamp = event_window[-1, 0]

                with Timer('Building event tensor'):
                    if args.compute_voxel_grid_on_cpu:
                        # pdb.set_trace()
                        event_tensor = events_to_voxel_grid(event_window, num_bins=model.num_bins, width=width,
                                                            height=height)
                        event_tensor = torch.from_numpy(event_tensor)
                    else:
                        # pdb.set_trace()
                        event_tensor = events_to_voxel_grid_pytorch(event_window, num_bins=model.num_bins,
                                                                    width=width, height=height, device=device)

                num_events_in_window = event_window.shape[0]
                reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window,
                                                    last_timestamp, frame_no, save_img_path)

                start_index += num_events_in_window
