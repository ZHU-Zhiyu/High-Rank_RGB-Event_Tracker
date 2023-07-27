import os
import sys
from os.path import join

import cv2
import numpy as np
from dv import AedatFile
from tqdm import tqdm

np.set_printoptions(suppress=True)

pair = {}


def get_start_frame(seq_name):
    return pair[seq_name]


def stack_event(index, root):
    match_file = 'pair.txt'
    with open(match_file, 'r') as f:
        for line in f.readlines():
            file, start_frame = line.split()
            pair[file] = int(start_frame) + 1
    root = root
    seq_name = root.split('/')[-1]
    img_path = os.path.join(root, 'img')
    frame_num = len(os.listdir(img_path))
    event_data = os.path.join(root, 'events.aedat4')
    stack_path = os.path.join(root, 'inter3_stack')
    start_frame = get_start_frame(seq_name)
    if not os.path.exists(stack_path):
        os.mkdir(stack_path)
    with AedatFile(event_data) as f:
        pic_shape = f['events'].size
        events = np.hstack([packet for packet in f['events'].numpy()])
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        event = np.vstack((timestamps, x, y, polarities))
        event = np.swapaxes(event, 0, 1)
        time_series = []
        count = 0
        for frame in f["frames"]:
            count += 1
            if count >= start_frame and count <= start_frame + frame_num:
                time_series.append(frame.timestamp_start_of_frame)
            else:
                continue
        event = event[event[:, 0] >= time_series[0]]
        event = event[event[:, 0] < time_series[-1]]
        deal_event(index, event, time_series, pic_shape, stack_path)


def process_event(pos_img, event, pic_shape):
    x, y, p = int(event[1]), int(event[2]), int(event[3])
    if 0 < x < pic_shape[1] and 0 < y < pic_shape[0]:
        if p == 1:
            pos_img[y][x] = 0
        else:
            pos_img[y][x] = 255


def deal_event(index,events, frame_timestamp, pic_shape, save_name):
    i = 1
    pos_img = np.full(pic_shape, 127, dtype=np.uint8)
    sub_index = 1
    sub_frame = np.linspace(frame_timestamp[0], frame_timestamp[1], 4)

    for event in tqdm(events, desc="{} Writing {} events ".format(index, save_name.split('/')[-2])):
        if event[0] >= frame_timestamp[i]:
            cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.jpg', pos_img)
            i = i + 1
            sub_frame = np.linspace(frame_timestamp[i - 1], frame_timestamp[i], 4)
            pos_img = np.full(pic_shape, 127, dtype=np.uint8)
            sub_index = 1
        elif event[0] < frame_timestamp[i]:
            if event[0] >= sub_frame[sub_index]:
                cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.jpg', pos_img)
                pos_img = np.full(pic_shape, 127, dtype=np.uint8)
                sub_index = sub_index + 1
            process_event(pos_img, event, pic_shape)

    cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(3) + '.jpg', pos_img)


def deal_event_single(index, events, frame_timestamp, pic_shape, save_name):
    i = 1
    pos_img = np.full(pic_shape, 127, dtype=np.uint8)
    sub_index = 1
    sub_frame = np.linspace(frame_timestamp[0], frame_timestamp[1], 2)

    for event in tqdm(events, desc="{} Writing {} events ".format(index, save_name.split('/')[-2])):
        if event[0] >= frame_timestamp[i]:
            cv2.imwrite(save_name + '/' + str(i).zfill(4) + '.jpg', pos_img)
            i = i + 1
            sub_frame = np.linspace(frame_timestamp[i - 1], frame_timestamp[i], 2)
            pos_img = np.full(pic_shape, 127, dtype=np.uint8)
            sub_index = 1
        elif event[0] < frame_timestamp[i]:
            if event[0] >= sub_frame[sub_index]:
                cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.jpg', pos_img)
                pos_img = np.full(pic_shape, 127, dtype=np.uint8)
                sub_index = sub_index + 1
            process_event(pos_img, event, pic_shape)

    cv2.imwrite(save_name + '/' + str(i).zfill(4) + '.jpg', pos_img)


s = int(sys.argv[1])
e = int(sys.argv[2])

file_name_list = []
# with open('eotb_train_split.txt', 'r')  as f:
#     a = [i.strip() for i in f.readlines()]
#     for line in a:
#         file_name_list.append(line)
with open('reshot_eotb_val_split.txt', 'r')  as f:
    a = [i.strip() for i in f.readlines()]
    for line in a:
        file_name_list.append(line)
for index,i in enumerate(sorted(file_name_list)[s:e]):
    data = os.path.join('/home/iccd/data/zjq/zjq_reshot/annotation/', i)
    # if i != 'drone1':
    #     continue
    if os.path.exists(join(data, 'inter3_stack')):
        if 3*len(os.listdir(join(data, 'img'))) == len(os.listdir(join(data, 'inter3_stack'))):
            continue
    stack_event(index, data)
