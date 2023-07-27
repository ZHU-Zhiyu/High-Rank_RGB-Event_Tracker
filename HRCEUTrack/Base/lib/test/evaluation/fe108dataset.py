import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class FE108Dataset(BaseDataset):

    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.fe108_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.fe108_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, sequence_name, "stack")
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        frame_list.sort(key=lambda f: int(f[-8:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        frames_event_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'voxel')
        frame_event_list = [frame for frame in os.listdir(frames_event_path) if frame.endswith(".mat")]
        frame_event_list.sort(key=lambda f: int(f[-8:-4]))
        frame_event_list = [os.path.join(frames_event_path, frame) for frame in frame_event_list]

        # event_img_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'dvs')
        # event_img_list = [frame for frame in os.listdir(event_img_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        # event_img_list.sort(key=lambda f: int(f[-8:-4]))
        # event_img_list = [os.path.join(event_img_path, frame) for frame in event_img_list]
        event_img_list = []

        return Sequence(sequence_name, frames_list, 'FE108', ground_truth_rect.reshape(-1, 4),
                        frame_event_list=frame_event_list, event_img_list=event_img_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'val' or split == 'train':
            with open('{}/{}.txt'.format(self.env_settings.dataspec_path, split)) as f:
                seq_ids = f.read().splitlines()
            sequence_list = [sequence_list[int(x)] for x in seq_ids]

        return sequence_list
