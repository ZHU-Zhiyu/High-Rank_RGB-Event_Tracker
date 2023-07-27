import math

from lib.models.ceutrack import build_ceutrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import copy

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import torch.nn.functional as F
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class CEUTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CEUTrack, self).__init__(params)
        network = build_ceutrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, event_template,  info: dict, idx=0):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr, crop_coor = sample_target(image, info['init_bbox'],
                                                                           self.params.template_factor,
                                                                           output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
        event_template = event_template.cuda()
        z = copy.deepcopy(event_template[:, 0])
        x, y = event_template[:, 1], event_template[:, 2]
        event_template[:, 0] = x
        event_template[:, 1] = y
        event_template[:, 2] = z
        x1, x2 = crop_coor[0] / 10, crop_coor[1] / 10
        y1, y2 = crop_coor[2] / 10, crop_coor[3] / 10
        x_range, y_range = x2-x1, y2-y1
        event_template[:, 0] = (event_template[:, 0]+0.5 - x1) / x_range
        event_template[:, 1] = (event_template[:, 1]+0.5 - y1) / y_range
        event_template[:, 2] = (event_template[:, 2]+0.5) / 19
        indices = (event_template[:, 0] >= 0) & (event_template[:, 0] <= 1) & (
                event_template[:, 1] >= 0) & (event_template[:, 1] <= 1)
        event_template = torch.index_select(event_template, dim=0, index=indices.nonzero().squeeze(1))

        event_template = event_template.unsqueeze(0).unsqueeze(0)
        if event_template.shape[2] >= 1024:
            event_template, _ = torch.topk(event_template, k=1024, dim=2)
            pad_len_temp = 0
        elif event_template.shape[2] < 1024:
            pad_len_temp = 1024 - event_template.shape[2]
        event_template = F.pad(event_template.transpose(-1, -2), (0, pad_len_temp), mode='constant', value=0)
        self.event_template = event_template
        # save states
        self.state = info['init_bbox']
        self.frame_id = idx
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, event_search, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr, crop_coor = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)   # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        event_search = event_search.cuda()
        z = copy.deepcopy(event_search[:, 0])
        x, y = event_search[:, 1], event_search[:, 2]
        event_search[:, 0] = x
        event_search[:, 1] = y
        event_search[:, 2] = z
        x1, x2 = crop_coor[0] / 10, crop_coor[1] / 10
        y1, y2 = crop_coor[2] / 10, crop_coor[3] / 10
        x_range, y_range = x2-x1, y2-y1
        event_search[:, 0] = (event_search[:, 0]+0.5 - x1) / x_range   # x voxel center
        event_search[:, 1] = (event_search[:, 1]+0.5 - y1) / y_range   # y voxel center
        event_search[:, 2] = (event_search[:, 2]+0.5) / 19                     # z voxel center (times)
        indices = (event_search[:, 0] >= 0) & (event_search[:, 0] <= 1) & (
                   event_search[:, 1] >= 0) & (event_search[:, 1] <= 1)
        event_search = torch.index_select(event_search, dim=0, index=indices.nonzero().squeeze(1))

        event_search = event_search.unsqueeze(0).unsqueeze(0)
        # event frame  need to keep same length  16,12-->20, 20
        if event_search.shape[2] < 4096:
            pad_len_search = 4096 - event_search.shape[2]
        else:
            event_search, _ = torch.topk(event_search, k=4096, dim=2)
            pad_len_search = 0

        event_search = F.pad(event_search.transpose(-1, -2), (0, pad_len_search), mode='constant', value=0)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, event_template=self.event_template,
                event_search=event_search,  ce_template_mask=self.box_mask_z,Track=True)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return CEUTrack
