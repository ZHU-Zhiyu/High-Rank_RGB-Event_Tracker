import random
import torch.utils.data
from pytracking import TensorDict
import numpy as np

def no_processing(data):
    return data

class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of a set of template frames and search frames, used to train the TransT model.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'template frames' and
    'search frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='interval'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template frames and the search frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the search frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_template_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_template_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + extra_template_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                                                              min_id=template_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                             max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_search_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * self.num_template_frames
            search_frame_ids = [1] * self.num_search_frames

        template_points, template_frames, template_anno, template_prev_anno,  meta_obj_template = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
        search_points, search_frames, search_anno, search_prev_anno,   meta_obj_search = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
        # search_points02 = []
        # template_points02 = []
        # for pnt in search_points:
        #     n2,_ = pnt.shape
        #     pn2 = np.random.randint(0,n2,[10000])
        #     pnt1 = pnt[pn2,:].astype(np.float)
        #     search_points02.append(pnt1)

        # for pnt in template_points:
        #     n2,_ = pnt.shape
        #     pn2 = np.random.randint(0,n2,[10000])
        #     pnt1 = pnt[pn2,:].astype(np.float)
        #     template_points02.append(pnt1)
        # print('size of search point:{}'.format(search_points[0].shape))
        # print('size of template point:{}'.format(template_points[0].shape))
        # print('min:{}, max{} of dim1 in search point'.format(search_points[0][:,0].min(), search_points[0][:,0].max()))
        # print('min:{}, max{} of dim2 in search point'.format(search_points[0][:,1].min(), search_points[0][:,1].max()))
        # print('min:{}, max{} of dim3 in search point'.format(search_points[0][:,2].min(), search_points[0][:,2].max()))
        # print('min:{}, max{} of dim4 in search point'.format(search_points[0][:,3].min(), search_points[0][:,3].max()))
        # print('size of search point:{}'.format(search_points[0].shape))
        data = TensorDict({'template_images': template_frames,
                           'template_anno': template_anno['bbox'],
                           'template_prev_anno': search_prev_anno['bbox'],
                           'search_images': search_frames,
                           'search_anno': search_anno['bbox'],
                           'search_prev_anno': search_prev_anno['bbox'],
                           'template_points':template_points,
                           'search_points':search_points})
        # print('===========srch_pnt:{}, template_pnt:{}'.format(search_points02[0].shape, template_points02[0].shape))
        # print('-------comming to data processing----------------')


        return self.processing(data)



class TransTSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_search_frames=num_search_frames, num_template_frames=num_template_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)