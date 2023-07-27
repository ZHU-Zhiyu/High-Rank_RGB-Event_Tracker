from os import device_encoding
from _collections import OrderedDict
from pytracking.tracker.transt.config import cfg
import cv2
import numpy as np
import torch

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params
        self.visdom = None
        self.curr_left_pad = None
        self.curr_top_pad = None
        self.pre_left_pad = None
        self.pre_top_pad = None
        self.scale_factor = None
        self.prev_bbox = None
    def predicts_segmentation_mask(self):
        return False


    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError


    def track(self, image, info: dict = None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        # print('------------- box content:{}, image shape :{}, mean of image:{}'.format(box, image.shape, image.mean(0).mean(0)))
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')

class SiameseTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans, Evt_point, prev_bbox=None):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        # left_pad = int(min(20, max(0., -context_xmin)))
        # top_pad = int(min(20,max(0., -context_ymin)))
        # right_pad = int(min(20,max(0., context_xmax - im_sz[1] + 1)))
        # bottom_pad = int(min(20,max(0., context_ymax - im_sz[0] + 1)))
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
        # print('im size [0]:{}, im size [1]:{}'.format(im_sz[0], im_sz[1]))
        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        self.curr_left_pad = left_pad
        self.curr_top_pad = top_pad
        if prev_bbox is not None:
            prev_bbox_out = torch.clone(prev_bbox)
            prev_bbox_out[:,0] = (prev_bbox_out[:,0] + left_pad - context_xmin)/ (context_xmax - context_xmin + 1e-6)
            prev_bbox_out[:,1] = (prev_bbox_out[:,1] + top_pad - context_ymin)/ (context_ymax - context_ymin + 1e-6)
            prev_bbox_out[:,2] = prev_bbox_out[:,2]/ (context_xmax - context_xmin + 1e-6)
            prev_bbox_out[:,3] = prev_bbox_out[:,3]/ (context_ymax - context_ymin + 1e-6)


        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        Evt_point[:,0] = left_pad/346 + Evt_point[:,0]
        Evt_point[:,1] = top_pad/260 + Evt_point[:,1]
        Evt_x_pos = Evt_point[:,0]
        Evt_y_pos = Evt_point[:,1]
        # print('shape of Evt_point:{}, max of crop x min:{}, max:{}, pad:{}, y min:{}, max:{}, pad:{}'.format(Evt_point.shape,  context_xmin, context_xmax, left_pad,  context_ymin, context_ymax, top_pad))
        flag_x1 = Evt_x_pos >= context_xmin/346
        flag_x2 = Evt_x_pos < context_xmax/346
        flag_y1 = Evt_y_pos >= context_ymin/260
        flag_y2 = Evt_y_pos < context_ymax/260

        flag = flag_x1 * flag_x2 * flag_y1 * flag_y2

        # flag = flag
        Evt_select = Evt_point[flag,:]

        Evt_select[:,0] = (Evt_select[:,0] - context_xmin/346) / (context_xmax/346 - context_xmin/346 + 1e-6)
        Evt_select[:,1] = (Evt_select[:,1] - context_ymin/260) / (context_ymax/260 - context_ymin/260 + 1e-6)
        # print('after-trans--------------min:{}, max{} of dim1 in search point'.format(Evt_select[:,0].min(), Evt_select[:,0].max()))
        # print('after-trans--------------min:{}, max{} of dim2 in search point'.format(Evt_select[:,1].min(), Evt_select[:,1].max()))
        # print('after-trans--------------min:{}, max{} of dim3 in search point'.format(Evt_select[:,2].min(), Evt_select[:,2].max()))
        # print('after-trans--------------min:{}, max{} of dim4 in search point'.format(Evt_select[:,3].min(), Evt_select[:,3].max()))


        n2,_ = Evt_select.shape
        if n2 == 0:
            Evt_point_crop = np.array([0,0,0,1])[np.newaxis,:]
            pn2 = np.random.randint(0,1,[10000])
            pnt1 = Evt_point_crop[pn2,:].astype(np.float)
        else:
            pn2 = np.random.randint(0,n2,[10000])
            pnt1 = Evt_select[pn2,:].astype(np.float)

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        pnt1 = torch.from_numpy(pnt1).to(im_patch.device)
        if prev_bbox is not None:
            return im_patch, pnt1[None,:,:], prev_bbox_out
        else:
            return im_patch, pnt1[None,:,:]