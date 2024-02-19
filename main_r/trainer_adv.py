import os
import numpy as np
from sys import exit
import torch
from typing import Union

from main_r.autoattack import AutoAttackMatGPUBase, AutoAttackMatGPUMTL
from main_r.fgsm import FGSMBase, FGSMMTL
from main_r.ifgsm import IFGSMBase, IFGSMMTL
from main_r.autosaga import SelfAttentionGradientAttackProto

class TrainerAdv():
    def __init__(self, model, tasks, train_dataloader, val_dataloader, criterion_dict, metric_dict,
                 attackVer, epsilon, att_criterion_dict, 
                 numSteps=20,  clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1., givenTask=None,
                 lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5,
                 print_iters=50, val_iters=200, save_iters=200):
        super(TrainerAdv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.startIter = 0
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
#         self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)
        
        self.tasks = tasks
        
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.criterion_dict = criterion_dict
        self.metric_dict = metric_dict
        
        self.loss_list = {}
        self.set_train_loss()
        
        self.print_iters = print_iters
        self.val_iters = val_iters
        self.save_iters = save_iters
        
        self.attackVer = attackVer
        self.epsilon = epsilon
        self.att_criterion_dict = att_criterion_dict
        self.numSteps = numSteps
        self.clipMin = clipMin
        self.clipMax = clipMax
        self.givenTask = givenTask
    
    def train(self, iters, no_ori=False, loss_lambda=None, savePath=None, reload=None):
        self.model.train()
        if reload and savePath is not None:
            self.load_model(savePath)
        if loss_lambda is None:
            loss_lambda = {task: 1 for task in self.tasks}

        for i in range(self.startIter, iters):
            data = self.train_step(loss_lambda, no_ori)
            self.train_adv(data, loss_lambda)

            if (i+1) % self.print_iters == 0:
                self.print_train_loss(i)
                self.set_train_loss()
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    self.save_model(i, savePath)
            if (i+1) % self.val_iters == 0:
                self.validate(i)
            
        # Reset loss list and the data iters
        self.set_train_loss()
        return
    
    def train_step(self, loss_lambda, no_ori):
        self.model.train()
        try:
            data = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            data = next(self.train_iter)
            
        if not no_ori:
            x = data['input'].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)

            loss = 0
            for task in self.tasks:
                y = data[task].to(self.device)
                mask = data[task+'_mask'].to(self.device) if task+'_mask' in data else None
                tloss = self.criterion_dict[task](output[task], y, mask)
                self.loss_list[task].append(tloss.item())
                loss += loss_lambda[task] * tloss
            self.loss_list['total'].append(loss.item())

            loss.backward()
            self.optimizer.step()
        return data
    
    def train_adv(self, data, loss_lambda):
        self.model.eval()
        advData = self.gen_adv(self.attackVer, self.epsilon, self.numSteps, self.clipMin, self.clipMax, data['input'], data, self.att_criterion_dict, self.givenTask)
        
        self.model.train()
        self.optimizer.zero_grad()
        advOutputs = self.model(advData)
        
        loss = 0
        for task in self.tasks:
            y = data[task].to(self.device)
            mask = data[task+'_mask'].to(self.device) if task+'_mask' in data else None
            tloss = self.criterion_dict[task](advOutputs[task], y, mask)
            self.loss_list[task].append(tloss.item())
            loss += loss_lambda[task] * tloss
        self.loss_list['total'].append(loss.item())
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return
    
    def validate(self, it=0):
        self.model.eval()
        loss_list = {}
        for task in self.tasks:
            loss_list[task] = []
        
        with torch.no_grad():
            for i, data in enumerate(self.val_dataloader):
                x = data['input'].to(self.device)
                output = self.model(x)

                for task in self.tasks:
                    y = data[task].to(self.device)
                    mask = data[task+'_mask'].to(self.device) if task+'_mask' in data else None
                    tloss = self.criterion_dict[task](output[task], y, mask)
                    self.metric_dict[task](output[task], y, mask)
                    loss_list[task].append(tloss.item())
        
        task_val_results = {}
        for task in self.tasks:
            avg_loss = np.mean(loss_list[task])
            val_results = self.metric_dict[task].val_metrics()
            print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
            print(val_results, flush=True)

        print('======================================================================', flush=True)
        
        torch.cuda.empty_cache()
        return
    
    # helper functions
    def set_train_loss(self):
        for task in self.tasks:
            self.loss_list[task] = []
        self.loss_list['total'] = []
        return
    
    def load_model(self, savePath):
        model_name = os.path.join(savePath, '_'.join([self.attackVer, str(self.givenTask)[:4], str(self.epsilon)]) + '.model')
        state = torch.load(model_name)
        self.startIter = state['iter'] + 1
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        return
    
    def save_model(self, it, savePath):
        state = {'iter': it,
                'state_dict': self.model.state_dict(),
                'layout': self.model.layout,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
        if hasattr(self.model, 'layout') and self.model.layout is not None:
            torch.save(state, os.path.join(savePath, '_'.join([self.attackVer, str(self.givenTask)[:4], str(self.epsilon)]) + '.model'))
        return
    
    def print_train_loss(self, it):
        # Function: Print loss for each task
        for task in self.tasks:
            if self.loss_list[task]:
                avg_loss = np.mean(self.loss_list[task])
            else:
                continue
            print('[Iter {} Task {}] Train Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
        print('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), np.mean(self.loss_list['total'])), flush=True)
        print('======================================================================', flush=True)
        return

    def gen_adv(self, attackVer, epsilon, numSteps, clipMin, clipMax, xData, yData, criterionDict, givenTask):
        ##################################### Attack Cases #####################################
        if attackVer == 'FGSM_Ori':
            advData = FGSMBase(self.device, xData, yData, self.model, criterionDict, 
                               epsilon, clipMin=clipMin, clipMax=clipMax, givenTask=givenTask)
        elif attackVer == 'FGSM_MTL':
            advData = FGSMMTL(self.device, xData, yData, self.model, criterionDict, 
                              epsilon, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'IFGSM_Ori':
            advData = IFGSMBase(self.device, xData, yData, self.model, criterionDict, 
                                epsMax=epsilon, alpha=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax, givenTask=givenTask)
        elif attackVer == 'IFGSM_MTL':
            advData = IFGSMMTL(self.device, xData, yData, self.model, criterionDict, 
                               epsMax=epsilon, alpha=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'AS_Ori':
            advData = SelfAttentionGradientAttackProto(self.device, xData, yData, self.model, self.tasks, criterionDict, 
                                                       epsMax=epsilon, epsStep=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'AS_MTL':
            advData = SelfAttentionGradientAttackProto(self.device, xData, yData, self.model, self.tasks, criterionDict, 
                                                       epsMax=epsilon, epsStep=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax, normalize=True)
                                               
        ##################### Cannot use because of missing criterionBase ################################       
        elif attackVer == 'AA_Ori':
            advData = AutoAttackMatGPUBase(self.device, xData, yData, self.model, criterionDict, 
                                           epsilon, etaStart=epsilon/numSteps, numSteps=numSteps, 
                                           clipMin=clipMin, clipMax=clipMax, givenTask=givenTask)
        elif attackVer == 'AA_MTL':
            criterionBase = {task: self.noAttackRe[task]['loss'] for task in self.tasks}
            advData = AutoAttackMatGPUMTL(self.device, xData, yData, self.model, criterionDict, criterionBase,
                                          epsilon, etaStart=epsilon/numSteps, numSteps=numSteps, 
                                          clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'AAR_Ori':
            advData = AutoAttackMatGPUBase(self.device, xData, yData, self.model, criterionDict, 
                                           epsilon, etaStart=epsilon/numSteps, numSteps=numSteps, 
                                           clipMin=clipMin, clipMax=clipMax, givenTask=givenTask, revised=True)
        elif attackVer == 'AAR_MTL':
            criterionBase = {task: self.noAttackRe[task]['loss'] for task in self.tasks}
            advData = AutoAttackMatGPUMTL(self.device, xData, yData, self.model, criterionDict, criterionBase,
                                          epsilon, etaStart=epsilon/numSteps, numSteps=numSteps, 
                                          clipMin=clipMin, clipMax=clipMax, revised=True)
        ##################################################################################################
        else:
            print('No such attack method!', flush=True)
            exit()
        return advData