import torch
import torch.nn as nn

import sys
sys.path.insert(0, '..')
from model.base import Bottleneck, BasicBlock
from model.deeplab_resnet import Deeplab_ResNet_Backbone
from data.pixel2pixel import ASPPHeadNode, SimpleHead

class IndNet(nn.Module):
    def __init__(self, cls_num, backbone:str='ResNet18', decoder:str='simple'):
        super(IndNet, self).__init__()
        if backbone == 'ResNet101':
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif backbone == 'ResNet18':
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif backbone == 'ResNet34':
            block = BasicBlock
            layers = [3, 4, 6, 3]
        
        self.backbone = Deeplab_ResNet_Backbone(block, layers=layers)
        if decoder == 'ASPP':
            self.head = ASPPHeadNode(512, cls_num) #### HEAD!!!
        elif decoder == 'simple':
            self.head = SimpleHead(512, cls_num)
            
    def forward(self, x):
        out = self.head(self.backbone(x))
        return out

class MTLNet(nn.Module):
    def __init__(self, tasks, cls_num:dict, backbone:str='ResNet18', decoder:str='simple', widen:float=1.0, deepen:int=0):
        super(MTLNet, self).__init__()
        self.tasks = tasks
        filt_sizes = [64, 128, 256, 512]
        if backbone == 'ResNet101':
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif backbone == 'ResNet18':
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif backbone == 'ResNet34':
            block = BasicBlock
            layers = [3, 4, 6, 3]
        
        filt_sizes = [int(x*widen) for x in filt_sizes]
        layers[-1] += deepen
        
        self.backbone = Deeplab_ResNet_Backbone(block, layers, filt_sizes)
        self.heads = nn.ModuleDict()
        for task in self.tasks:
            if decoder == 'ASPP':
                self.heads[task] = ASPPHeadNode(filt_sizes[-1], cls_num[task])
            elif decoder == 'simple':
                self.heads[task] = SimpleHead(filt_sizes[-1], cls_num[task])
            
    def forward(self, x):
        feat = self.backbone(x)
        out = {task: self.heads[task](feat) for task in self.tasks}
        return out
    
