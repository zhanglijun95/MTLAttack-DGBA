import os
import pickle
import numpy as np
import warnings
import torch
import collections
import csv
from typing import Union

warnings.filterwarnings("ignore", category=UserWarning) 
from main_r.autoattack import AutoAttackMatGPUBase, AutoAttackMatGPUMTL, AutoAttackMatGPUSignSum
from main_r.fgsm import FGSMBase, FGSMMTL, FGSMSignSum
from main_r.ifgsm import IFGSMBase, IFGSMMTL, IFGSMSignSum
from main_r.autosaga import SelfAttentionGradientAttackProto

class Attacker():
    def __init__(self, model, tasks:list, valDataloader, criterionDictBase, criterionDictMTL, metricDict, noAttackRe:dict, saveCSV):
        ########################################################################
        ### 1. model should be in device already and has been loaded weights ###
        ### 2. criterion's reduction should be none for MTL attack           ###
        ### 3. savePath is for saving csv, include the csv name already      ###
        ########################################################################
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.tasks = tasks
        self.saveCSV = saveCSV
        
        self.valDataloader = valDataloader
        self.criterionDictBase = criterionDictBase
        self.criterionDictMTL = criterionDictMTL
        self.metricDict = metricDict
        self.noAttackRe = noAttackRe
        
        if len(self.tasks) == 3: # NYUv2
            self.__metrics_prop = {'mIoU': False, 'Pixel Acc': False, 
                                  'Angle Mean': True, 'Angle Median': True, 'Angle 11.25': False, 'Angle 22.5': False, 'Angle 30': False,
                                  'abs_err': True,'rel_err': True,'sigma_1.25': False,'sigma_1.25^2': False,'sigma_1.25^3': False,
                                  'loss': True, 'acc.': False} # lower the better
        elif len(self.tasks) == 5: # Taskonomy
            self.__metrics_prop = {'err': True, 'abs_err': True, 'cosine_similarity': False, 'key_err': True, 'edge_err': True, 'loss': True, 'acc.': False}
        
        self.__fields = ['version', 'epsilon', 'others', 'avg. acc.', 'avg loss'] + [task[:4]+' acc.' for task in self.tasks] + [task[:4]+' loss' for task in self.tasks]
        self.__fields += [key for key in self.__metrics_prop if not key in ['loss', 'acc.']]
        
    def attack(self, attackVer, epsilon, clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1., givenTask=None, **kwargs):
        starter = {'epsilon': epsilon, 'others': str(kwargs)}
        if 'Ori' in attackVer:
            starter['version'] = 'ind_'+givenTask[:4] if givenTask else 'total_loss'
            criterionDict = self.criterionDictBase
        elif 'MTL' in attackVer:
            starter['version'] = 'mtl'
            criterionDict = self.criterionDictMTL
        elif 'SignSum' in attackVer:
            starter['version'] = 'sign_sum'
            criterionDict = self.criterionDictMTL
        numSteps = kwargs['numSteps'] if 'numSteps' in kwargs else 10 #10 for taskonomy AA & AS, 20 for others
        
        # check if already attacked with the current start['version'] and epsilon
        if self.check_csv(starter['version'], epsilon):
            return
        
        val_loss = {task: [] for task in self.tasks}
        for i, data in enumerate(self.valDataloader):
            xData = data['input']
            yData = data
            
            advData = self.gen_adv(attackVer, epsilon, numSteps, clipMin, clipMax, xData, yData, criterionDict, givenTask)
        
            #### test acc. / loss for advData
            with torch.no_grad():
                advOutputs = self.model(advData)
                for task in self.tasks:
                    y = data[task].to(self.device)
                    mask = data[task+'_mask'].to(self.device) if task+'_mask' in data else None
                    self.metricDict[task](advOutputs[task], y, mask)
                    val_loss[task].append(criterionDict[task](advOutputs[task], y, mask).mean().item())

        #### print val results
        task_val_results = {}
        for task in self.tasks:
            val_results = self.metricDict[task].val_metrics()
            val_results['loss'] = np.mean(val_loss[task])
            task_val_results[task] = val_results
            print('Task {}: {}'.format(task[:4], val_results), flush=True)
        print('-'*100,flush=True)
        
        #### compare with baseline
        rel_results = self.compute_rel(task_val_results)
        #### write to csv
        self.save_csv(rel_results, starter)
        torch.cuda.empty_cache()
        return
        
    def gen_adv(self, attackVer, epsilon, numSteps, clipMin, clipMax, xData, yData, criterionDict, givenTask):
        ##################################### Attack Cases #####################################
        if attackVer == 'FGSM_Ori':
            advData = FGSMBase(self.device, xData, yData, self.model, criterionDict, 
                               epsilon, clipMin=clipMin, clipMax=clipMax, givenTask=givenTask)
        elif attackVer == 'FGSM_MTL':
            advData = FGSMMTL(self.device, xData, yData, self.model, criterionDict, 
                              epsilon, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'FGSM_SignSum':
            advData = FGSMSignSum(self.device, xData, yData, self.model, criterionDict, 
                              epsilon, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'IFGSM_Ori':
            advData = IFGSMBase(self.device, xData, yData, self.model, criterionDict, 
                                epsMax=epsilon, alpha=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax, givenTask=givenTask)
        elif attackVer == 'IFGSM_MTL':
            advData = IFGSMMTL(self.device, xData, yData, self.model, criterionDict, 
                               epsMax=epsilon, alpha=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'IFGSM_SignSum':
            advData = IFGSMSignSum(self.device, xData, yData, self.model, criterionDict, 
                               epsMax=epsilon, alpha=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'AS_Ori':
            advData = SelfAttentionGradientAttackProto(self.device, xData, yData, self.model, self.tasks, criterionDict, 
                                                       epsMax=epsilon, epsStep=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'AS_MTL':
            advData = SelfAttentionGradientAttackProto(self.device, xData, yData, self.model, self.tasks, criterionDict, 
                                                       epsMax=epsilon, epsStep=epsilon/numSteps, numSteps=numSteps, clipMin=clipMin, clipMax=clipMax, normalize=True)
        elif attackVer == 'AA_Ori':
            advData = AutoAttackMatGPUBase(self.device, xData, yData, self.model, criterionDict, 
                                           epsilon, etaStart=epsilon/numSteps, numSteps=numSteps, 
                                           clipMin=clipMin, clipMax=clipMax, givenTask=givenTask)
        elif attackVer == 'AA_MTL':
            criterionBase = {task: self.noAttackRe[task]['loss'] for task in self.tasks}
            advData = AutoAttackMatGPUMTL(self.device, xData, yData, self.model, criterionDict, criterionBase,
                                          epsilon, etaStart=epsilon/numSteps, numSteps=numSteps, 
                                          clipMin=clipMin, clipMax=clipMax)
        elif attackVer == 'AA_SignSum':
            advData = AutoAttackMatGPUSignSum(self.device, xData, yData, self.model, criterionDict,
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
        else:
            print('No such attack method!', flush=True)
            exit()
        return advData
        
    def compute_rel(self, task_val_results):
        rel_results = {}
        for task in self.tasks:
            tmp = 0
            for metric in task_val_results[task]:
                value = task_val_results[task][metric]
                baseline = self.noAttackRe[task][metric]
                if metric == 'loss':
                    rel_results[task[:4]+' loss'] = round((baseline-value)/baseline, 6)
                    continue
                rel_results[metric] = value
                if self.__metrics_prop[metric]:
                    tmp += (baseline-value)/baseline
                else:
                    tmp += (value-baseline)/baseline
            tmp /= len(task_val_results[task])-1 # len(task_val_results[task])-1 = #M
            rel_results[task[:4]+' acc.'] = round(tmp, 6)  

        avg_loss, avg_acc = 0, 0
        for task in self.tasks:
            avg_loss += rel_results[task[:4]+' loss']
            avg_acc += rel_results[task[:4]+' acc.']
        rel_results['avg. acc.'] = round(avg_acc/len(self.tasks), 4)
        rel_results['avg loss'] = round(avg_loss/len(self.tasks), 4)
        print('Rel. Results: {}'.format(rel_results),flush=True)
        print('-'*100,flush=True)
        return rel_results
    
    def save_csv(self, rel_results, starter):
        if not os.path.exists(self.saveCSV):
            with open(self.saveCSV, 'w') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(self.__fields)
                
        with open(self.saveCSV, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = self.__fields)
            writer.writerow(starter | rel_results)
        return
    
    def check_csv(self, version, epsilon):
        if not os.path.exists(self.saveCSV):
            return False
        with open(self.saveCSV, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if line[0] == version and float(line[1]) == epsilon:
                    return True
        return False