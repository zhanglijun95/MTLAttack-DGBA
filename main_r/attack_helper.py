import torch
import numpy as np

#Function for computing the model gradient for total loss or ind. loss
def GetModelGradient(device, xK, yK:dict, model, criterionDict, task=None): 
    xK.requires_grad = True
    
    #Pass the inputs through the model 
    if task != None:
        output = model(xK, task=task)
        xK.grad = None
        model.zero_grad()
        mask = yK[task+'_mask'].to(device) if task+'_mask' in yK else None
        cost = criterionDict[task](output, yK[task].to(device), mask)
    else:
        outputs = model(xK) 
        xK.grad = None
        model.zero_grad()
        cost = 0.
        for task in criterionDict:
            mask = yK[task+'_mask'].to(device) if task+'_mask' in yK else None
            cost += criterionDict[task](outputs[task], yK[task].to(device), mask)
            
    cost.backward()
    xKGrad = xK.grad
    return xKGrad.detach()

#Function for computing the sum of normalized model gradient
def GetModelGradientMTL(device, xK, yK:dict, model, criterionDict):
    xK.requires_grad = True
    outputs = model(xK)
    
    direction = torch.zeros(xK.shape).to(device)
    for task in criterionDict:
        xK.grad = None
        model.zero_grad()
        mask = yK[task+'_mask'].to(device) if task+'_mask' in yK else None
        loss = criterionDict[task](outputs[task], yK[task].to(device), mask)
        loss.mean().backward(retain_graph=True)
        xGrad = xK.grad.detach()
        direction += xGrad/loss.detach()[:,None,None,None]
    return direction

#Function for computing the sum of signed model gradient
def GetModelGradientSignSum(device, xK, yK:dict, model, criterionDict):
    xK.requires_grad = True
    outputs = model(xK)
    
    direction = torch.zeros(xK.shape).to(device)
    for task in criterionDict:
        xK.grad = None
        model.zero_grad()
        mask = yK[task+'_mask'].to(device) if task+'_mask' in yK else None
        loss = criterionDict[task](outputs[task], yK[task].to(device), mask)
        loss.mean().backward(retain_graph=True)
        xGrad = xK.grad.detach()
        direction += xGrad.sign()
    return direction