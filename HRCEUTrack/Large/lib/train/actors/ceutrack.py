from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class CEUTrackActor(BaseActor):
    """ Actor for training CEUTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1
        assert len(data['template_event']) == 1
        assert len(data['search_event']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        template_event = data['template_event'][0].view(-1, *data['template_event'].shape[2:])
        search_event = data['search_event'][0].view(-1, *data['search_event'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            event_template=template_event,
                            event_search=search_event,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            loss = 'Nan'
            if return_status:
                status = {"Loss/total": 0,
                        "Loss/giou": 0,
                        "Loss/l1": 0,
                        "Loss/location": 0,
                        "IoU": 0}
                return loss, status
            else:
                return loss

            # raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        rank_loss = self.loss_rank(pred_dict,gt_dict['search_anno'], gt_dict['template_anno'])
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + rank_loss*0.8
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def _random_permute(self,matrix):
        # matrix = random.choice(matrix)
        b, c, h, w = matrix.shape
        idx = [  torch.randperm(c).to(matrix.device) for i in range(b)]
        idx = torch.stack(idx, dim=0)[:, :, None, None].repeat([1,1,h,w])
        # idx = torch.randperm(c)[None,:,None,None].repeat([b,1,h,w]).to(matrix.device)
        matrix01 = torch.gather(matrix, 1, idx)
        return matrix01
    def crop_flag(self, flag, global_index_s, global_index_t,H1 = 64, H2 = 256):
        B,Ls = global_index_s.shape
        B, Lt = global_index_t.shape
        B,C,L1,L2 = flag.shape
        flag_t = flag[:,:,:H1,:]
        flag_s = flag[:,:,H1:,:]

        flag_t = torch.gather(flag_t,2,global_index_t[:,None,:,None].repeat([1,C,1,L2]).long())
        flag_s = torch.gather(flag_s,2,global_index_s[:,None,:,None].repeat([1,C,1,L2]).long())
        flag = torch.cat([flag_t, flag_s], dim = 2)

        flag_t = flag[:,:,:,:H1]
        flag_s = flag[:,:,:,H1:]
        flag_t = torch.gather(flag_t,3,global_index_t[:,None,None,:].repeat([1,C,int(Ls+Lt),1]).long())
        flag_s = torch.gather(flag_s,3,global_index_s[:,None,None,:].repeat([1,C,int(Ls+Lt),1]).long())
        flag = torch.cat([flag_t, flag_s], dim = 3)
        B, C, L11, L12 = flag.shape
        try:
            assert(L11 == int(Lt + Ls))
            assert(L12 == int(Lt + Ls))
        except:
            print('L11:{}, L12:{}, L1:{}, L2:{}'.format(L11, L12, L1, L2))
        return flag
    def crop_fusion(self, flag, attn, global_index_s, global_index_t,H1 = 64, H2 = 256 ):
        flag = self.crop_flag(flag=flag, global_index_s=global_index_s, global_index_t=global_index_t)
        B,C,L1,L2 = flag.shape
        Ba, Ca, La, La2 = attn.shape
        _,idx1 = flag.mean(dim=3,keepdim=False).sort(dim=2,descending=True)
        # print('shape of flag:{}, idx1:{}'.format(flag.shape, idx1[:,:,:32,None].repeat([1,Ca,1,L2]).shape))
        flag = torch.gather(flag,2,idx1[:,:,:32,None].repeat([1,C,1,L2]).long())
        attn = torch.gather(attn,2,idx1[:,:,:32,None].repeat([1,Ca,1,L2]).long())
        _,idx2 = flag.mean(dim=2,keepdim=False).sort(dim=2,descending=True)
        flag = torch.gather(flag,3,idx2[:,:,None,:32].repeat([1,C,32,1]).long())
        attn = torch.gather(attn,3,idx2[:,:,None,:32].repeat([1,Ca,32,1]).long())
        return attn * flag

    def loss_rank(self, outputs, targetsi, temp_annoi=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        attn = outputs['attn']
        # print('attn shape:{}'.format(attn.shape))
        attn1 = torch.cat([attn[:,:,114:344,57:114], attn[:,:,114:344,344:]],dim=3)
        attn1 = attn1.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        attn2 = torch.cat([attn[:,:,344:,:57], attn[:,:,344:,114:344]],dim=3)
        attn2 = attn2.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        # print('attn1 shape:{},attn2 shape:{}, attn:{}'.format(attn1.shape,attn2.shape,attn.shape))

        # attn = self._random_permute(attn)
        # attn = attn[:,:,:,:]
        # B1, C1, H1, W1 = attn.shape
        # global_index_s = outputs['out_global_s']
        # global_index_t = outputs['out_global_t']
        # try:
        #     assert((global_index_s.shape[1] + global_index_t.shape[1])== int(H1/2))
        # except:
        #     print('Falut,shape of attn:{}, s:{}, t:{}'.format(attn.shape,global_index_s.shape, global_index_t.shape ))

        # H1 = int(64)
        # H2 = int(256)
        # l_t = int(math.sqrt(64))
        # l_s = int(math.sqrt(256))

        # temp_anno = temp_annoi[0,:,:]
        # targets = targetsi[0,:,:]
        # r_s = torch.arange(l_s).to(temp_anno.device)
        # r_t = torch.arange(l_t).to(temp_anno.device)
        # r_t = r_t[None,:].repeat([B1,1])

        # cx, cy, w, h = temp_anno[:,0:1], temp_anno[:,1:2], temp_anno[:,2:3], temp_anno[:,3:4]
        # cx *= l_t
        # cy *= l_t
        # w *= l_t
        # h *= l_t
        # flagx_01 = r_t >= cx - w/2
        # flagx_02 = r_t <= cx + w/2
        # flagy_02 = r_t >= cy - h/2
        # flagy_01 = r_t <= cy + h/2
        # flagx = flagx_01.float()*flagx_02.float()
        # flagy = flagy_01.float()*flagy_02.float()
        # flagx = flagx[:,None,:].repeat([1,l_t,1])
        # flagy = flagy[:,:,None].repeat([1,1,l_t])
        # flag = flagx*flagy
        # flagt = flag.reshape([B1, H1])

        # cx, cy, w, h = targets[:,0:1], targets[:,1:2], targets[:,2:3], targets[:,3:4]
        # cx *= l_s
        # cy *= l_s
        # w *= l_s
        # h *= l_s
        # flagx_01 = r_s >= cx - w/2
        # flagx_02 = r_s <= cx + w/2
        # flagy_02 = r_s >= cy - h/2
        # flagy_01 = r_s <= cy + h/2
        # flagx = flagx_01.float()*flagx_02.float()
        # flagy = flagy_01.float()*flagy_02.float()
        # flagx = flagx[:,None,:].repeat([1,l_s,1])
        # flagy = flagy[:,:,None].repeat([1,1,l_s])
        # flag = flagx*flagy

        # flags = flag.reshape([B1, H2])

        # flag = torch.cat([flagt, flags], dim=1)
        # flag_total = flag[:,:,None].repeat([1,1,int(H1+H2)]) * flag[:,None,:].repeat([1,int(H1+H2),1])
        # attn1 = self.crop_fusion(flag_total[:,None,:,:], attn, global_index_s, global_index_t)
        attn = torch.cat([attn1, attn2],dim=1)
        B, C, H, W = attn.shape
        # _,s1,_ = torch.svd(attn1.reshape([B*C, H, W]))

        _,s1,_ = torch.svd(attn.reshape([B*C, H, W]))

        s01 = torch.abs(s1 - 1)

        return torch.mean(s01)
