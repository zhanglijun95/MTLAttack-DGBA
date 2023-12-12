import sys
sys.path.append('../TreeMTL/')
import numpy as np
import random
import os
import argparse
from scipy import stats
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main_r.trainer_adv import TrainerAdv
from main_r.utils import *

from main.layout import Layout
from main.algorithms import enum_layout_wo_rdt, init_S, coarse_to_fined
from main.auto_models import MTSeqBackbone, MTSeqModel
from main.models import Deeplab_ASPP_Layout

from data.nyuv2_dataloader_adashare import NYU_v2
from data.taskonomy_dataloader_adashare import Taskonomy
from data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', action='store', dest='dataroot', default='/work/lijunzhang_umass_edu/data/policymtl/data/', help='datasets directory')
# parser.add_argument('--weightroot', action='store', dest='weightroot', default='/work/lijunzhang_umass_edu/data/multibranch/checkpoint/', help='weights directory')
parser.add_argument('--ckpt_dir', action='store', dest='ckpt_dir', default='./ckpt/MTLAttacker/', help='checkpoints directory')

parser.add_argument('--data', action='store', dest='data', default='NYUv2', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=64, type=int, help='dataset batch size')
parser.add_argument('--backbone', action='store', dest='backbone', default='resnet34', help='backbone model')
parser.add_argument('--layout_idx', action='store', dest='layout_idx', default=0, type=int, help='layout index')
parser.add_argument('--attackVer', action='store', dest='attackVer', default='FGSM_Ori', help='attack method')
parser.add_argument('--eps', action='store', dest='eps', default=8, type=int, help='attack epsilon')
parser.add_argument('--givenTask', action='store', dest='givenTask', default=-1, type=int, help='is ind attack and its task')

parser.add_argument('--seed', action='store', dest='seed', default=10, type=int, help='seed')
parser.add_argument('--no_ori', action='store', dest='no_ori', default=0, type=int, help='whether to use original data')
parser.add_argument('--reload', action='store', dest='reload', default=0, type=int, help='whether to reload')
parser.add_argument('--val_iters', action='store', dest='val_iters', default=2000, type=int, help='frequency of validation')
parser.add_argument('--print_iters', action='store', dest='print_iters', default=200, type=int, help='frequency of print')
parser.add_argument('--save_iters', action='store', dest='save_iters', default=200, type=int, help='frequency of model saving')
parser.add_argument('--lr', action='store', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay_lr_freq', action='store', dest='decay_lr_freq', default=4000, type=int, help='frequency of lr decay')
parser.add_argument('--decay_lr_rate', action='store', dest='decay_lr_rate', default=0.5, type=float, help='rate of lr decay')
parser.add_argument('--total_iters', action='store', dest='total_iters', default=20000, type=int, help='total iterations')

args = parser.parse_args()
print(args, flush=True)
print('='*60, flush=True)
assert torch.cuda.is_available()

################################### Gen Params ################################
dataroot = os.path.join(args.dataroot, args.data)
# if args.data == 'NYUv2' and args.backbone == 'resnet34':
#     weightroot = os.path.join(args.weightroot, args.data, 'verify_1118')
# elif args.data == 'Taskonomy' and args.backbone == 'resnet34':
#     weightroot = os.path.join(args.weightroot, args.data, 'verify_0123')
# else:
#     print('Wrong dataset and backbone combination!')
#     exit()
advroot = os.path.join(args.ckpt_dir, '_'.join([args.data, args.backbone]), args.attackVer.split('_')[0], str(args.layout_idx), str(args.no_ori))
if not os.path.exists(advroot):
    os.makedirs(advroot)

################################### Load Data #####################################
np.random.seed(args.seed)
if args.data == 'NYUv2':
    tasks = ['segment_semantic','normal','depth_zbuffer']
    cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
    T = len(tasks)
    
    dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
    trainDataloader = DataLoader(dataset, args.bz, shuffle=True)
    
    dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
    valDataloader = DataLoader(dataset, args.bz, shuffle=True)
    
    criterionDictBase = {task: NYUCriterions(task) for task in tasks}
    criterionDictMTL = {task: NYUCriterions(task, reduction='none') for task in tasks}
    metricDict = {task: NYUMetrics(task) for task in tasks}
    
    clipMin = torch.tensor([-104.00698793, -116.66876762, -122.67891434])
    clipMax = torch.tensor([255-104.00698793, 255-116.66876762, 255-122.67891434])
elif args.data == 'Taskonomy': 
    tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
    cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
    T = len(tasks)
    
    dataset = Taskonomy(dataroot, 'test_small', crop_h=224, crop_w=224)
    valDataloader = DataLoader(dataset, args.bz, shuffle=False)

    criterionDictBase = {task: TaskonomyCriterions(task, dataroot) for task in tasks}
    criterionDictMTL = {task: TaskonomyCriterions(task, dataroot, reduction='none') for task in tasks}
    metricDict = {task: TaskonomyMetrics(task, dataroot) for task in tasks}
    
    clipMin = torch.tensor([-104.00698793, -116.66876762, -122.67891434])
    clipMax = torch.tensor([255-104.00698793, 255-116.66876762, 255-122.67891434])
else:
    print('Wrong dataset!')
    exit()

print('Finish Data Loading', flush=True)

########################## Params from Backbone #################################
if args.backbone == 'resnet34':
    prototxt = '../TreeMTL/models/deeplab_resnet34_adashare.prototxt'
    coarse_B, fined_B = 5, 17
    feature_dim = 512
    mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
##### TODO: ADD MORE BACKBONE TYPE ##### 
else:
    print('Wrong backbone!')
    exit()
print('Finish Preparing Params', flush=True)

####################### Transfer Layouts ##############################
layout_list = [] 
S0 = init_S(T, coarse_B) # initial state
L = Layout(T, coarse_B, S0) # initial layout
layout_list.append(L)

enum_layout_wo_rdt(L, layout_list)

layout = layout_list[args.layout_idx]
print('Coarse Layout:', flush=True)
print(layout, flush=True)
layout = coarse_to_fined(layout, fined_B, mapping)
print('Fined Layout:', flush=True)
print(layout, flush=True)
print('Finish Layout Emueration and Selection', flush=True)
print('='*60, flush=True)

################################ Generate Model ##################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.data == 'NYUv2' and args.backbone == 'resnet34':
    torch.manual_seed(args.seed)
    model = Deeplab_ASPP_Layout(layout, cls_num).to(device)
else: # auto-models
    if max(mapping[max(mapping)]) != fined_B:
        print('Wrong mapping for the given backbone model because of inconsistent number of blocks.')
        exit()
    torch.manual_seed(args.seed)
    model = MTSeqModel(prototxt, layout=layout, feature_dim=feature_dim, cls_num=cls_num).to(device)

# load weights
# state = torch.load(os.path.join(weightroot, str(args.layout_idx), '_'.join(tasks) + '.model'))
# model.load_state_dict(state['state_dict']) 
# model.eval()
print('Finish Model Generation', flush=True)

################################### Train #####################################
print('###################### Start Train ######################', flush=True)
attCriterionDict = criterionDictBase if 'Ori' in args.attackVer else criterionDictMTL
no_ori = True if args.no_ori == 1 else False
reload = True if args.reload == 1 else False
givenTask = tasks[args.givenTask] if args.givenTask != -1 else None
trainer = TrainerAdv(model, tasks, trainDataloader, valDataloader, criterionDictBase, metricDict,
                    args.attackVer, args.eps, attCriterionDict, 
                    numSteps=10,  clipMin=clipMin, clipMax=clipMax, givenTask=givenTask,
                    lr=args.lr, decay_lr_freq=args.decay_lr_freq, decay_lr_rate=args.decay_lr_rate, 
                    print_iters=args.print_iters, val_iters=args.val_iters, save_iters=args.save_iters)
trainer.train(args.total_iters, no_ori=no_ori, savePath=advroot, reload=reload)
print('###################### Finish Train ######################', flush=True)