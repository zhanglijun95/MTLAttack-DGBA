import torch
import numpy as np
from typing import Union
from .attack_helper import GetModelGradient, GetModelGradientMTL, GetModelGradientSignSum

# original FGSM
def FGSMBase(device, xData, yData:dict, model, criterionDict, 
             epsilon, clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1., 
             givenTask=None):
    model.eval() #Change model to evaluation mode for the attack
    
    x = xData.to(device)
    y = yData #A dict for all the tasks' labels
        
    if isinstance(clipMin, torch.Tensor):
        clipMin = clipMin.to(device)
        clipMax = clipMax.to(device)
        
    xGrad = GetModelGradient(device, x, y, model, criterionDict, givenTask) 
    # adversarial attack
    advX = x + epsilon*torch.sign(xGrad)
    advX = torch.clamp(advX.permute(0,2,3,1), min=clipMin, max=clipMax).permute(0,3,1,2)
    torch.cuda.empty_cache()
    return advX

# FGSMMTL
def FGSMMTL(device, xData, yData:dict, model, criterionDict,
            epsilon, clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1.):
    model.eval() #Change model to evaluation mode for the attack
    
    x = xData.to(device)
    y = yData #A dict for all the tasks' labels
        
    if isinstance(clipMin, torch.Tensor):
        clipMin = clipMin.to(device)
        clipMax = clipMax.to(device)
        
    xGrad = GetModelGradientMTL(device, x, y, model, criterionDict) 
    # adversarial attack
    advX = x + epsilon*torch.sign(xGrad)
    advX = torch.clamp(advX.permute(0,2,3,1), min=clipMin, max=clipMax).permute(0,3,1,2)
    torch.cuda.empty_cache()
    return advX

# FGSMSignSum
def FGSMSignSum(device, xData, yData:dict, model, criterionDict,
            epsilon, clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1.):
    model.eval() #Change model to evaluation mode for the attack
    
    x = xData.to(device)
    y = yData #A dict for all the tasks' labels
        
    if isinstance(clipMin, torch.Tensor):
        clipMin = clipMin.to(device)
        clipMax = clipMax.to(device)
        
    xGrad = GetModelGradientSignSum(device, x, y, model, criterionDict) 
    # adversarial attack
    advX = x + epsilon*torch.sign(xGrad)
    advX = torch.clamp(advX.permute(0,2,3,1), min=clipMin, max=clipMax).permute(0,3,1,2)
    torch.cuda.empty_cache()
    return advX