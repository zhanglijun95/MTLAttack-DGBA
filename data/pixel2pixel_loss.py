import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataCriterions(nn.Module):
    def __init__(self, task):
        super(DataCriterions, self).__init__()
        self.task = task
        
    def define_loss(self):
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss(reduction=self.reduction)
        
    def seg_loss(self, pred, gt):
        new_shape = pred.shape[-2:]
        prediction = pred.contiguous()
        gt = F.interpolate(gt.float(), size=new_shape).squeeze(dim=1).contiguous()
        loss = self.cross_entropy(prediction, gt.long())
        if self.reduction == 'none':
            if self.cross_entropy.weight == None:
                loss = loss.sum(dim=(1,2))/(gt!=self.cross_entropy.ignore_index).sum(dim=(1,2))
            else: 
                loss = loss.sum(dim=(1,2))/self.cross_entropy.weight[gt.long()].sum(dim=(1,2))
        return loss
    
    def sn_loss(self, pred, gt, mask=None):
        new_shape = pred.shape[-2:]
        prediction = pred.contiguous()
        gt = F.interpolate(gt.float(), size=new_shape).contiguous()
        labels = (gt.max(dim=1)[0] < 255)
        if mask != None: 
            gt_mask = F.interpolate(mask.float(), size=new_shape).contiguous()
            labels = torch.logical_and(labels, gt_mask[:, 0].int() == 1)
        
        prediction = F.normalize(prediction)
        gt = F.normalize(gt)
        if self.reduction == 'mean':
            loss = 1 - self.cosine_similiarity(prediction, gt)[labels].mean()
        elif self.reduction == 'none':
            loss = 1 - (self.cosine_similiarity(prediction, gt)*labels).sum(dim=(1,2))/labels.sum(dim=(1,2))
        return loss
    
    def depth_loss(self, pred, gt, mask=None):
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape) 
        if mask != None:
            gt_mask = F.interpolate(mask.float(), size=new_shape)
            binary_mask = (gt != 255) * (gt_mask.int() == 1)
        else:
            binary_mask = (torch.sum(gt, dim=1) > 3 * 1e-5).unsqueeze(1).cuda()
        if self.reduction == 'mean':
            prediction = pred.masked_select(binary_mask)
            gt = gt.masked_select(binary_mask)
            loss = self.l1_loss(prediction, gt)
        elif self.reduction == 'none':
            loss = (self.l1_loss(pred, gt)*binary_mask).sum(dim=(1,2,3))/binary_mask.sum(dim=(1,2,3))
        return loss
    
    def keypoint_edge_loss(self, pred, gt):
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape)
        binary_mask = gt != 255
        if self.reduction == 'mean':
            prediction = pred.masked_select(binary_mask)
            gt = gt.masked_select(binary_mask)
            loss = self.l1_loss(prediction, gt)
        elif self.reduction == 'none':
            loss = (self.l1_loss(pred, gt)*binary_mask).sum(dim=(1,2,3))/binary_mask.sum(dim=(1,2,3))
        return loss
        
    def forward(self, pred, gt, mask=None):
        if self.task == 'segment_semantic':
            return self.seg_loss(pred, gt)
        elif self.task == 'normal':
            return self.sn_loss(pred, gt, mask)
        elif self.task == 'depth_zbuffer':
            return self.depth_loss(pred, gt, mask)
        elif self.task == 'keypoints2d' or self.task == 'edge_texture':
            return self.keypoint_edge_loss(pred, gt)
        else:
            print('Wrong task for the criterion!', flush=True)
        
class TaskonomyCriterions(DataCriterions):
    def __init__(self, task, dataroot, reduction='mean'):
        super(TaskonomyCriterions, self).__init__(task)
        self.reduction = reduction
        if self.task == 'segment_semantic':
            self.num_seg_cls = 17
        self.define_loss(dataroot)
        
    def define_loss(self, dataroot):
        super(TaskonomyCriterions, self).define_loss()
        weight = torch.from_numpy(np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))).cuda().float()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255, reduction=self.reduction)
        
class CityScapesCriterions(DataCriterions):
    def __init__(self, task, reduction='mean'):
        super(CityScapesCriterions, self).__init__(task)
        self.reduction = reduction
        if self.task == 'segment_semantic':
            self.num_seg_cls = 19
        self.define_loss()
        
    def define_loss(self):
        super(CityScapesCriterions, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1, reduction=self.reduction)
        
class NYUCriterions(DataCriterions):
    def __init__(self, task, reduction='mean'):
        super(NYUCriterions, self).__init__(task)
        self.reduction = reduction
        if self.task == 'segment_semantic':
            self.num_seg_cls = 40
        self.define_loss()
        
    def define_loss(self):
        super(NYUCriterions, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255, reduction=self.reduction)