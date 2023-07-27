import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.models.tracking.transt_utils import Feature_extraction, gether_neibour
from torchvision.utils import save_image
import random
from knn_cuda import KNN
import math
def point2img(points, H, W):
    B, C, P = points.shape
    x_pos = points[0,:,0]
    y_pos = points[0,:,1]
    
    x_pos = points[0,:,0]*H
    x_pos = x_pos.long()

    y_pos = points[0,:,1]*W
    y_pos = y_pos.long()
    img = torch.zeros(3,H,W).float().to(points.device)
    polar = points[0,:,3]
    x_pos = torch.clamp(x_pos,0, H-1)
    y_pos = torch.clamp(y_pos,0, W-1)
    img[0,y_pos,x_pos] = torch.clamp(polar,0).float()
    img[2,y_pos,x_pos] = -torch.clamp(polar,max=0).float()
    return img


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.backbone = backbone
        # self.Raw_event_projector = event_projector()
        self.Evt_project = Feature_extraction()
        self.idx = 0
        self.motion_estimate = nn.Sequential(nn.Linear(96, 16, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(16, 2, bias=True),)
        self.motion_fea = nn.Sequential(nn.Linear(18, 64, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(64, 32, bias=True),)
        self.Ouput_conv = nn.Conv2d(32,2,kernel_size=4, stride = 4)
        self.knn_1 = KNN(k=1, transpose_mode=False)

    def forward(self, search, template, Evt_srch, Evt_temp, img_ratio,prev_bbox):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels
               - REvt: batched event tensor, of shape [batch_size x 4 x NEvent]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        # print('type of search:{}, template:{}, Evt_srch:{}, Evt_temp:{}'.format(type(search), type(template), type(Evt_srch), type(Evt_temp)))
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
            
        Srch_pnt = self.Evt_project(Evt_srch.permute([0,2,1]).float(), out_num = 4096)
        Temp_pnt = self.Evt_project(Evt_temp.permute([0,2,1]).float(), out_num = 1024)

        feature_search, pos_search, point_serch_fea, img_serch_fea = self.backbone.train_forward(search, Raw_point = Srch_pnt, ratio = img_ratio)
        feature_template, pos_template, point_temp_fea, img_temp_fea = self.backbone.train_forward(template, Raw_point = Temp_pnt, ratio = img_ratio)

        mon_attn, Evt_pos, diff_= self.box_encode(prev_bbox= prev_bbox.float(), src_search=point_serch_fea[3],Evt_pos=point_serch_fea[0],orig_Evt= Evt_srch.permute([0,2,1])[:,:4,:].float())
        src_search, mask_search= feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        # hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(self.mask_fusion(src_search, mon_attn)), mask_search, pos_template[-1], pos_search[-1])
        loss_back = self.calculate_flow_loss(Evt_pos, Evt_srch.permute([0,2,1]).float())

        # a = random.randint(0,10000)

        # img_sch = src_search[:1,:,:,:].mean(1,keepdim=True).cpu()
        # img_temp = src_template[:1,:,:,:].mean(1,keepdim=True).cpu()
        # img_sch = nn.functional.interpolate(img_sch, [512,512], mode='bicubic')
        # img_temp = nn.functional.interpolate(img_temp, [256,256], mode='bicubic')
        # save_image(img_sch,'/data/zzy/ablation/imgs_transformer/Search_fea_%08d.jpg'%a)
        # save_image(img_temp,'/data/zzy/ablation/imgs_transformer/Temp_fea_%08d.jpg'%a)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],'Temp_img':img_temp_fea,
                    'Temp_pnt':point_temp_fea,'Search_img':img_serch_fea,'Search_pnt':point_serch_fea, 'loss_back':loss_back+diff_*1e-1, 'mon_attn':mon_attn}
        # Output = outputs_class[-1].reshape([16,16,16,2])
        # print('shape of pred_logits.mean_pos0:{},pred_logits.mean_pos64:{}, pred_logits.mean_pos128:{}'.format(Output[:,:4,:4,:].mean(0).mean(0).mean(0).detach().cpu().numpy(), 
        #                                                                                                 Output[:,4:12,4:12,:].mean(0).mean(0).mean(0).detach().cpu().numpy(), 
        #                                                                                                 Output[:,12:,12:,:].mean(0).mean(0).mean(0).detach().cpu().numpy()))
        return out

    def track(self, search, e,prev_bbox):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)

        Srch_pnt = self.Evt_project(e.permute([0,2,1]).float(), out_num = 4096)
        features_search, pos_search, point_serch_fea, img_serch_fea = self.backbone(search, Raw_point = Srch_pnt)
        encode_fea,_,_ = self.box_encode(prev_bbox= prev_bbox.float(), src_search=point_serch_fea[3],Evt_pos=point_serch_fea[0],orig_Evt= e.permute([0,2,1])[:,:4,:].float())
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None

        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search*F.softmax(encode_fea, dim=1)[:,:1,:,:]), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def calculate_flow_loss(self, Evt_pos, input_pos):
        """
        Evt_pos in B, 2, H, W
        input_pos in B, 4, L
        """
        threshold = 2e-2
        # print('input_pos.shape:{}, Evt_pos.shape:{}'.format(input_pos.shape,Evt_pos.shape))
        assert(input_pos.shape[1] == 4)
        assert(Evt_pos.shape[1] == 2)
        _, idx = input_pos[:,2,:].sort(-1,descending=True)

        input_pos = torch.gather(input_pos[:,:2,:], 2, idx[:,None,:].repeat(1,2,1))

        input_pos = input_pos[:,:,:2000]
        B, C, H, W = Evt_pos.shape
        Evt_pos = Evt_pos.reshape([B,C,H*W])
        _, indx = self.knn_1(input_pos.contiguous(),Evt_pos.contiguous())
        #  B, P1, n1, C
        Evt_pos_target = gether_neibour(input_pos, indx, Evt_pos, 1)
        Evt_pos_target = Evt_pos_target[:,:,0,:2].permute([0,2,1])

        _, indx = self.knn_1(Evt_pos.contiguous(),input_pos.contiguous())
        #  B, P1, n1, C
        input_pos_target = gether_neibour(Evt_pos, indx, input_pos, 1)
        input_pos_target = input_pos_target[:,:,0,:2].permute([0,2,1])

        loss1 = torch.sum(torch.abs(input_pos_target - input_pos[:,:2,:]), 1)
        loss2 = torch.sum(torch.abs(Evt_pos_target - Evt_pos[:,:2,:]), 1)
        mask1 = loss1 < threshold
        mask2 = loss2 < threshold

        loss1 = loss1 * mask1.float()
        loss2 = loss2 * mask2.float()

        return torch.mean(loss1) + torch.mean(loss2)

    def mask_fusion(self, sem, mon,ratio=1/8):
        
        # B, C, H, W =sem.shape
        # patch_num = int(1/ratio)
        # patch_size = int(H * ratio)
        # flag = torch.rand([B,1,1,patch_num,patch_num],device = sem.device) > 0.2
        # flag = flag.float()
        # flag = flag.repeat([1,patch_size, patch_size, 1, 1])
        # flag = flag.permute([0,3,1,4,2]).reshape(B,H,W)
        # flag1 = flag[:,None,:,:].repeat(1,C,1,1)
        # flag_n1 = 1- flag1

        # mask = mon*flag1 + flag_n1

        return sem*F.softmax(mon, dim=1)[:,:1,:,:]


    def box_encode(self,prev_bbox, src_search, Evt_pos, orig_Evt ):
        """
        prev_bbox in shape of cx cy w h, [B,4]
        orig_Evt in shape of b, 4, p
        """
        # print('shape of Evt_pos:{}, src_search:{}'.format(Evt_pos.shape, src_search.shape))
        # print('shape of Prev_bbox:{},src_search:{}, Evt_pos:{}, orig_Evt:{}'.format(prev_bbox.shape, src_search.shape, Evt_pos.shape, orig_Evt.shape))
        b, c, l = Evt_pos.shape
        h1, w1 = int(math.sqrt(l)), int(math.sqrt(l))
        Evt_pos = Evt_pos.reshape([b,c,h1,w1])
        # b, _, h1, w1 = Evt_pos.shape
        b, c, h, w = src_search.shape
        # assert(h1==h)
        # assert(w1==w)
        assert(orig_Evt.shape[1]==4)
        # Evt_pos = Evt_pos.reshape([b,4, w*h])
        # src_search_ = src_search.reshape([b,c, w*h])
        vc = self.motion_estimate(Evt_pos[:,4:,:,:].permute([0,2,3,1]))

        min_t = orig_Evt[:,2,:].min(-1,keepdim=False)[0][:,None,None]
        max_t = orig_Evt[:,2,:].max(-1,keepdim=False)[0][:,None,None]
        norm_t = torch.abs(Evt_pos[:,2,:,:]- min_t)/ (torch.abs(max_t - min_t) + 1e-6)

        # flow in shape of B, 2, H, W
        diff_ = vc.permute([0,3,1,2])
        diff_ = torch.mean(torch.abs(diff_[:,:,1:,:] -diff_[:,:,:-1,:])) + torch.mean(torch.abs(diff_[:,:,:,1:] -diff_[:,:,:,:-1]))
        diff_ = diff_
        flow = vc.permute([0,3,1,2])*norm_t[:,None,:,:]

        Evt_pos = Evt_pos[:,:2,:,:] + flow
        # print('shape of Evt_pos:{}'.format(Evt_pos.shape))
        encode_Evt = self.Evt_project.Embedding_layer01.sin_cos_encoding(Evt_pos)

        prev_bbox_1 = prev_bbox[:,:2] - prev_bbox[:,2:]/2
        prev_bbox_2 = prev_bbox[:,:2] + prev_bbox[:,2:]/2
        encode_bbox = self.Evt_project.Embedding_layer01.sin_cos_encoding(torch.stack([prev_bbox_1, prev_bbox_2],2)[:,:,:,None])
        encode_bbox = torch.cat([encode_bbox[:,:,0,0],encode_bbox[:,:,1,0]], dim = 1)

        encode_fea = torch.cat([encode_Evt, encode_bbox[:,:,None,None].repeat([1,1,h1,w1])],dim=1)

        encode_fea = self.motion_fea(encode_fea.permute([0,2,3,1]))
        encode_fea = self.Ouput_conv(encode_fea.permute([0,3,1,2]))
        # print('shape of encode_feature:{}'.format(encode_fea.shape))
        return encode_fea, Evt_pos, diff_
    def template(self, z, e):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        # zf, pos_template = self.backbone(z, Raw_point = e)
        Temp_pnt = self.Evt_project(e.permute([0,2,1]).float(), out_num = 1024)
        zf, pos_template, point_temp_fea, img_temp_fea = self.backbone(z, Raw_point = Temp_pnt)
        # self.zf = zf
        self.zf = zf
        self.pf = point_temp_fea
        self.pos_template = pos_template

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # print('shape of src_logits in label:{}'.format(src_logits.shape))

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_motion_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'mon_attn' in outputs
        src_logits = outputs['mon_attn']
        B, C, H, W = src_logits.shape
        src_logits = src_logits.permute([0,2,3,1]).reshape([B, H*W, C])
        # print('shape of src_logits:{}'.format(src_logits.shape))

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_mon_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_mon_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'loss_motion_labels': self.loss_motion_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        losses.update(self.get_loss('loss_motion_labels', outputs, targets, indices, num_boxes_pos))
        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5, 'loss_motion_labels': 8}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
