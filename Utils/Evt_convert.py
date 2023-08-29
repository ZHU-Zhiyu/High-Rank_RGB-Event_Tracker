from ast import Delete
import os
import h5py
from dv import AedatFile
import numpy as np
import gc
from tqdm import tqdm

gc.collect()

train_names = []

Path_FE108 = None
Path_out = None

train_name_file ='Please set here to the path to train.txt'
match_file = 'Please set here to the path to pair.txt'
aedat_file = 'Please set here to the path to events.aedat4'
save_file = 'Please set here to the path to save h5 file'

with open(train_name_file, 'r') as f:
    for line in f.readlines():
        # if os.path.exists('/data1/zhu_19/saved_mat/'+line.replace("\n", "")):
        train_names.append(line.replace("\n", ""))

            
start_frame_ = {}
with open(match_file, 'r') as f:
    for line in f.readlines():
        file, start_frame = line.split()
        start_frame_[file] = int(start_frame) + 1

        
for name in train_names[::-1]:
    start_frame = start_frame_[name]

    print(name)
    gc.collect()
    with AedatFile(os.path.join(aedat_file,  name, 'events.aedat4')) as f1:
        output_event = os.path.join(aedat_file,  name, 'events.aedat4')
        if os.path.exists(output_event) == False:
            print('name:{}'.format(name))
            gt_bbox = np.loadtxt(os.path.join(aedat_file,  name, 'groundtruth_rect.txt'),delimiter=',')
            
            fam_num, length = gt_bbox.shape
            events = [packet for packet in f1['events'].numpy()]
            events = np.hstack(events)
            timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
            events = np.vstack((  x, y, timestamps, polarities))
            events = np.swapaxes(events, 0, 1)
            frames = f1['frames']

            fam = [i for i in frames]

            fam_start = [ fam[i].timestamp_start_of_frame for i in range(len(fam))]

            H,W = fam[0].size

            gt_bbox[:,0] = gt_bbox[:,0] / H
            gt_bbox[:,1] = gt_bbox[:,1] / W
            gt_bbox[:,2] = gt_bbox[:,2] / H
            gt_bbox[:,3] = gt_bbox[:,3] / W
            # print('Norm box max:{},min:{}'.format(gt_bbox.max(),gt_bbox.min()))
            gt_bbox = gt_bbox.clip(0,1)

            # self.start_frame = 710
            fam_start = fam_start[start_frame-1: start_frame+fam_num]

            # all_frame_start[name] = fam_start

            # # self.GT = gt_bbox
            # all_GT[name] = gt_bbox
            events = events.astype(np.float_)
            events[:,0] = events[:,0]/H
            events[:,1] = events[:,1]/W
            frame_length = len(fam_start)
            n = 10
            fam_start_mid01 = fam_start[ int(frame_length//n) + 2 ]

            
            eT = events[:,2]
            eW = events[:,0]
            eH = events[:,1]
            eA = events[:,3]
            # split_event = {}
            flag1 = eT <= fam_start_mid01
            Evt_curr = events[flag1,:]
            eTCurr = Evt_curr[:,2]

            curr_flag = int(frame_length//n)

            output_file = h5py.File(output_event, 'w')
        
            for i in tqdm(range(fam_num)):
                assert(i < fam_num)
                if i+1 > curr_flag:
                    curr_flag = int(curr_flag +  int(frame_length//n))
                    fam_start_mid01 = fam_start[ min(curr_flag, frame_length - 1)]
                    fam_start_mid02 = fam_start[ max(curr_flag - int(frame_length//n)-1 , 0)]
                    flag1 = eT <= fam_start_mid01
                    flag2 = eT >= fam_start_mid02
                    Evt_curr = events[flag1*flag2,:]
                    eTCurr = Evt_curr[:,2]

                s_time = fam_start[i]
                e_time = fam_start[i + 1]
                flag1 = eTCurr >= s_time
                flag2 = eTCurr <= e_time
                flag = flag1*flag2
                event_temp = Evt_curr[flag,:]


                output_file.create_dataset(str(i),data = event_temp)

            output_file.close()
        f1.close()