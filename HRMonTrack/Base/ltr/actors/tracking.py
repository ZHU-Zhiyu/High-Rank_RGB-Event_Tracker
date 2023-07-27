from . import BaseActor
import torch
import numpy as np

class TranstActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data, ratio):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        h, w = data['search_images'][0][0].shape
        prev_bbox =  data['search_prev_anno']
        prev_bbox[:,0] /= w
        prev_bbox[:,1] /= h
        prev_bbox[:,2] /= w
        prev_bbox[:,3] /= h
        prev_bbox[:,0] += prev_bbox[:,2] / 2
        prev_bbox[:,1] += prev_bbox[:,3] / 2

        noise = (torch.rand_like(prev_bbox, device = prev_bbox.device).float()-0.5)*1.2

        prev_bbox = prev_bbox *(1+noise)

        outputs = self.net(data['search_images'], data['template_images'], data['search_points'], data['template_points'], ratio, prev_bbox =prev_bbox)

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)
        temp_anno = transfer_data(data['template_anno'], images=data['template_images'])
        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets, temp_anno)
        weight_dict = self.objective.weight_dict
        # loss_modal = [ (outputs['Temp_pnt'][i] - outputs['Temp_img'][i]).abs().mean() + (outputs['Temp_pnt'][i] - outputs['Temp_img'][i]).abs().mean() for i in range(len(outputs['Temp_pnt']))]
        # loss_modal = [ (torch.mean((outputs['Temp_pnt'][i] - outputs['Temp_img'][i])**2)+ torch.mean((outputs['Temp_pnt'][i] - outputs['Temp_img'][i])**2))*((i+1)/float(len(outputs['Temp_pnt'])))**1.5 for i in range(len(outputs['Temp_pnt']))]
        # loss_modal = torch.sum(torch.stack(loss_modal, dim=0))
        loss_back = outputs['loss_back']
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) + ((outputs['Temp_pnt'] - outputs['Temp_img']).abs().mean() + (outputs['Search_pnt'] - outputs['Search_img']).abs().mean())*1e1
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) + loss_back

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'Loss_mon_ce': loss_dict['loss_mon_ce'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats

def transfer_data(orig_box, images):
    targets =[]

    for i in range(len(orig_box)):
        h, w =images[i][0].shape
        target_origin = orig_box[i]
        target = {}
        target_origin = target_origin.reshape([1,-1])
        target_origin[0][0] += target_origin[0][2] / 2
        target_origin[0][0] /= w
        target_origin[0][1] += target_origin[0][3] / 2
        target_origin[0][1] /= h
        target_origin[0][2] /= w
        target_origin[0][3] /= h
        target['boxes'] = target_origin
        label = np.array([0])
        label = torch.tensor(label, device=orig_box.device)
        target['labels'] = label
        targets.append(target)
    
    return targets

