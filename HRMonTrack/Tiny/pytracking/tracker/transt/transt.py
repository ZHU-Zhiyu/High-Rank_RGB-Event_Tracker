from pytracking.tracker.base import BaseTracker, SiameseTracker
import torch
import torch.nn.functional as F
import math
import time
import numpy as np
from pytracking.tracker.transt.config import cfg
import torchvision.transforms.functional as tvisf

class TransT(SiameseTracker):

    multiobj_mode = 'parallel'

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, Evt_point, info: dict) -> dict:
        hanning = np.hanning(16)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        # Initialize network
        self.initialize_features()
        # The DiMP network
        self.net = self.params.net
        # Time initialization
        tic = time.time()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0]+bbox[2]/2,
                                    bbox[1]+bbox[3]/2])
        self.size = np.array([bbox[2], bbox[3]])
        self.prev_bbox = torch.from_numpy(np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2, bbox[2], bbox[3]])).float()[None,:]
        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop, evt_cloud  = self.get_subwindow(image, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average,Evt_point = Evt_point)
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
        
        # print(dir(self.net))

        self.net.template(z_crop, evt_cloud)
        out = {'time': time.time() - tic}
        return out

    def _bbox_clip(self, cx, cy, width, height, boundary):
        
        px1 = cx - width/ 2
        py1 = cy - height/ 2
        px2 = cx + width/ 2
        py2 = cy + height/ 2
        px1 = max(0, min(px1, boundary[0]))
        px2 = max(0, min(px2, boundary[0]))
        py1 = max(0, min(py1, boundary[1]))
        py2 = max(0, min(py2, boundary[1]))

        cx = (px1 + px2)/ 2
        cy = (py1 + py2)/ 2
        width = max(10, min(px2 - px1, boundary[0]))
        height = max(10, min(py2 - py1, boundary[1]))
        # cx = max(0, min(cx, boundary[1]))
        # cy = max(0, min(cy, boundary[0]))
        # width = max(10, min(width, boundary[1]))
        # height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def track(self, image,Evt_point,  info: dict = None) -> dict:
        # w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        # h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        x_crop, evt_cloud, prev_bbox_out = self.get_subwindow(image, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average,Evt_point = Evt_point, prev_bbox = self.prev_bbox)
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)
        print('----------------------------input of prev_bbox_out:{}'.format(prev_bbox_out))
        outputs = self.net.track(x_crop, evt_cloud, prev_bbox_out.float().to(x_crop.device))
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])

        # def change(r):
        #     return np.maximum(r, 1. / r)
        #
        # def sz(w, h):
        #     pad = (w + h) * 0.5
        #     return np.sqrt((w + pad) * (h + pad))
        # # pred_box:cx,cy,w,h
        # # scale penalty
        # s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
        #              (sz(self.size[0]/s_x, self.size[1]/s_x)))
        #
        # # aspect ratio penalty
        # r_c = change((self.size[0]/self.size[1]) /
        #              (pred_bbox[2, :]/pred_bbox[3, :]))
        # penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        # pscore = penalty * score

        # window penalty
        pscore = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        # pscore = score
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:,best_idx]

        bbox = bbox * s_x
        cx1 = bbox[0] + self.center_pos[0] - s_x/2
        cy1 = bbox[1] + self.center_pos[1] - s_x/2

        # smooth bbox
        # no penaty
        width1 = bbox[2]
        height1 = bbox[3]

        img_shape = image.shape[:2]
        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx1, cy1, width1,
                                                height1, [img_shape[1], img_shape[0]])
        self.prev_bbox = torch.tensor([cx,cy,width,height],device = self.prev_bbox.device)[None,:].float()
        # print('Before / after clipping cx:{}/{}, cy:{}/{}, w:{}/{}, h:{}/{}'.format(cx1, cx, cy1, cy, width1, width, height1, height))
        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        print('output box:{}'.format(bbox))
        out = {'target_bbox': bbox,
               'best_score': pscore}
        return out