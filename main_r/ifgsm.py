import torch
import numpy as np
from typing import Union
from .attack_helper import GetModelGradient, GetModelGradientMTL, GetModelGradientSignSum

# original IFGSM
def IFGSMBase(device, xData, yData:dict, model, criterionDict, 
             epsMax, alpha, numSteps, 
             clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1., givenTask=None):
    model.eval() #Change model to evaluation mode for the attack
    
    x = xData.to(device)
    xOriMax = x + epsMax
    xOriMin = x - epsMax
    y = yData #A dict for all the tasks' labels
    if isinstance(clipMin, torch.Tensor):
        clipMin = clipMin.to(device)
        clipMax = clipMax.to(device)
    
    for i in range(numSteps):
        xGrad = GetModelGradient(device, x, y, model, criterionDict, givenTask) 
        # adversarial attack
        advX = x + alpha*torch.sign(xGrad)
        advX = torch.min(advX, xOriMax)
        advX = torch.max(advX, xOriMin)
        advX = torch.clamp(advX.permute(0,2,3,1), min=clipMin, max=clipMax).permute(0,3,1,2)
        x = advX.detach()
        torch.cuda.empty_cache()
    return advX

# IFGSMMTL
def IFGSMMTL(device, xData, yData:dict, model, criterionDict,
            epsMax, alpha, numSteps,
            clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1.):
    model.eval() #Change model to evaluation mode for the attack
    
    x = xData.to(device)
    xOriMax = x + epsMax
    xOriMin = x - epsMax
    y = yData #A dict for all the tasks' labels
    if isinstance(clipMin, torch.Tensor):
        clipMin = clipMin.to(device)
        clipMax = clipMax.to(device)
    
    for i in range(numSteps):
        xGrad = GetModelGradientMTL(device, x, y, model, criterionDict) 
        # adversarial attack
        advX = x + alpha*torch.sign(xGrad)
        advX = torch.min(advX, xOriMax)
        advX = torch.max(advX, xOriMin)
        advX = torch.clamp(advX.permute(0,2,3,1), min=clipMin, max=clipMax).permute(0,3,1,2)
        x = advX.detach()
        torch.cuda.empty_cache()
    return advX

# iFGSMSignSum
def IFGSMSignSum(device, xData, yData:dict, model, criterionDict,
            epsMax, alpha, numSteps,
            clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1.):
    model.eval() #Change model to evaluation mode for the attack
    
    x = xData.to(device)
    xOriMax = x + epsMax
    xOriMin = x - epsMax
    y = yData #A dict for all the tasks' labels
    if isinstance(clipMin, torch.Tensor):
        clipMin = clipMin.to(device)
        clipMax = clipMax.to(device)
    
    for i in range(numSteps):
        xGrad = GetModelGradientSignSum(device, x, y, model, criterionDict) 
        # adversarial attack
        advX = x + alpha*torch.sign(xGrad)
        advX = torch.min(advX, xOriMax)
        advX = torch.max(advX, xOriMin)
        advX = torch.clamp(advX.permute(0,2,3,1), min=clipMin, max=clipMax).permute(0,3,1,2)
        x = advX.detach()
        torch.cuda.empty_cache()
    return advX