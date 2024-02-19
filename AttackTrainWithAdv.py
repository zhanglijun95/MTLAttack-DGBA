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

from main_r.Attacker import Attacker
from main_r.no_attack import no_attack_results
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
parser.add_argument('--no_ori', action='store', dest='no_ori', default=0, type=int, help='whether to use original data')

parser.add_argument('--data', action='store', dest='data', default='NYUv2', help='experiment dataset')
parser.add_argument('--batch_size', action='store', dest='bz', default=64, type=int, help='dataset batch size')
parser.add_argument('--backbone', action='store', dest='backbone', default='resnet34', help='backbone model')
parser.add_argument('--layout_idx', action='store', dest='layout_idx', default=0, type=int, help='layout index')
parser.add_argument('--attackVer', action='store', dest='attackVer', default='FGSM_Ori', help='attack method')
parser.add_argument('--eps', action='store', dest='eps', default=8, type=int, help='attack epsilon')
parser.add_argument('--givenTask', action='store', dest='givenTask', default=-1, type=int, help='is ind attack and its task')

args = parser.parse_args()
print(args, flush=True)
print('='*60, flush=True)
assert torch.cuda.is_available()

################################### Gen Params ################################
dataroot = os.path.join(args.dataroot, args.data)
if args.data == 'NYUv2' and args.backbone == 'resnet34':
    weightroot = os.path.join(args.ckpt_dir, '_'.join([args.data, args.backbone]), args.attackVer.split('_')[0], str(args.layout_idx), str(args.no_ori))
# elif args.data == 'NYUv2' and args.backbone == 'resnet34_2':
#     weightroot = os.path.join(args.weightroot, args.data, 'verify_0411')
# elif args.data == 'Taskonomy' and args.backbone == 'resnet34':
#     weightroot = os.path.join(args.weightroot, args.data, 'verify_0123')
##### TODO: ADD MORE WEIGHTROOT FOR DIFFERENT DATA AND BACKBONE #####
else:
    print('Wrong dataset and backbone combination!')
    exit()
csvroot = os.path.join(weightroot, 'csv')
if not os.path.exists(csvroot):
    os.makedirs(csvroot)
noattackroot = os.path.join(weightroot, 'no_attack')
if not os.path.exists(noattackroot):
    os.makedirs(noattackroot)

################################### Load Data #####################################
if args.data == 'NYUv2':
    tasks = ['segment_semantic','normal','depth_zbuffer']
    cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
    T = len(tasks)
    
    dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
    valDataloader = DataLoader(dataset, args.bz, shuffle=True)
    criterionDictBase = {task: NYUCriterions(task) for task in tasks}
    criterionDictMTL = {task: NYUCriterions(task, reduction='none') for task in tasks}
    metricDict = {task: NYUMetrics(task) for task in tasks}
    
    clipMin = torch.tensor([-104.00698793, -116.66876762, -122.67891434])
    clipMax = torch.tensor([255-104.00698793, 255-116.66876762, 255-122.67891434])
# elif args.data == 'Taskonomy': 
#     tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
#     cls_num = {'segment_semantic': 17, 'normal': 3, 'depth_zbuffer': 1, 'keypoints2d': 1, 'edge_texture': 1}
#     T = len(tasks)
    
#     dataset = Taskonomy(dataroot, 'test_small', crop_h=224, crop_w=224)
#     valDataloader = DataLoader(dataset, args.bz, shuffle=False)

#     criterionDictBase = {task: TaskonomyCriterions(task, dataroot) for task in tasks}
#     criterionDictMTL = {task: TaskonomyCriterions(task, dataroot, reduction='none') for task in tasks}
#     metricDict = {task: TaskonomyMetrics(task, dataroot) for task in tasks}
    
#     clipMin = torch.tensor([-104.00698793, -116.66876762, -122.67891434])
#     clipMax = torch.tensor([255-104.00698793, 255-116.66876762, 255-122.67891434])
else:
    print('Wrong dataset!')
    exit()

print('Finish Data Loading', flush=True)

########################## Params from Backbone #################################
if 'resnet34' in args.backbone:
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
    model = Deeplab_ASPP_Layout(layout, cls_num).to(device)
else: # auto-models
    if max(mapping[max(mapping)]) != fined_B:
        print('Wrong mapping for the given backbone model because of inconsistent number of blocks.')
        exit()
    model = MTSeqModel(prototxt, layout=layout, feature_dim=feature_dim, cls_num=cls_num).to(device)

# load weights
givenTask = tasks[args.givenTask] if args.givenTask != -1 else None
state = torch.load(os.path.join(weightroot, '_'.join([args.attackVer, str(givenTask)[:4], str(args.eps)]) + '.model'))
model.load_state_dict(state['state_dict']) 
model.eval()
print('Finish Model Generation', flush=True)

########################## No Attack Baseline ############################
no_attack_file = os.path.join(noattackroot, '_'.join([args.attackVer, str(givenTask)[:4], str(args.eps)])+'.no')
if os.path.exists(no_attack_file):
    noAttackRe = load_pickle(no_attack_file)
else:
    noAttackRe = no_attack_results(device, model, tasks, valDataloader, criterionDictBase, metricDict)
    save_pickle(noAttackRe, no_attack_file)
print('No Attack Results: {}'.format(noAttackRe), flush=True)
print('Finish Load Baseline', flush=True)

################################### Attack #####################################
print('###################### Start Attack ######################', flush=True)
saveCSV = os.path.join(csvroot, '_'.join([args.attackVer, str(givenTask)[:4], str(args.eps)]) +'.csv')
attacker = Attacker(model, tasks, valDataloader, criterionDictBase, criterionDictMTL, metricDict, noAttackRe, saveCSV)

attack = args.attackVer.split('_')[0]
for case in ['Ori', 'MTL']:
    newAttackVer = '_'.join([attack, case])
    print('Case {}'.format(newAttackVer), flush=True)
    if case == 'Ori':
        ### for ind. loss
        for task in tasks: 
            print('Task {} Epsilon {}'.format(task, args.eps), flush=True)
            attacker.attack(newAttackVer, args.eps, clipMin, clipMax, givenTask=task)
    ### for total loss or MTL attack
    print('Task All Epsilon {}'.format(args.eps), flush=True)
    attacker.attack(newAttackVer, args.eps, clipMin, clipMax)
print('###################### Finish Attack ######################', flush=True)