import torch
import numpy as np
from typing import Union

# Main attack method, takes in a list of models and a clean data loader
# Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttackProto(device, xClean, yClean:dict, mtl_model, tasks:list, criterionDict:dict,
                                     epsMax, epsStep, numSteps,
                                     clipMin:Union[float,torch.Tensor]=0., clipMax:Union[float,torch.Tensor]=1., alphaLearningRate=1e1, fittingFactor=50, decay=0,
                                     normalize=False): #alphaLearningRate=1e10
    mtl_model.eval()
    mtl_model.to(device)
    
    xAdv = xClean.to(device)  # Set the initial adversarial samples
    numSamples = xClean.shape[0] # Get the total number of samples to attack
    xShape = xClean.shape[1:]  # Get the shape of the input (there may be easier way to do this)
    
    xOridata = xClean.detach().to(device)
    xOriMax = xOridata + epsMax
    xOriMin = xOridata - epsMax
    
    if isinstance(clipMin, torch.Tensor):
        clipMin = clipMin.to(device)
        clipMax = clipMax.to(device)

    alpha = torch.ones(len(tasks), numSamples, xShape[0], xShape[1],xShape[2]).to(device)  # alpha for every model and every sample
    xGradientCumulativeB = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2]).to(device)

    for i in range(0, numSteps):
        # print("Running step", i)
        # print("---------------------------------------------")
        dCdX = torch.zeros(len(tasks), numSamples, xShape[0], xShape[1], xShape[2]).to(device)
        dFdX = torch.zeros(numSamples, xShape[0], xShape[1],xShape[2]).to(device)  # Change to the math here to take in account all objecitve functions
        
        for m, task in enumerate(tasks):
            dCdX[m] = NativeGradient(device, mtl_model, task, xAdv, yClean, criterionDict, normalize) # gradient for one specific task
        
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2]).to(device)
        for m in range(len(tasks)):
            xGradientCumulative = xGradientCumulative + alpha[m] * dCdX[m]
            
        xAdvStepOne = xAdv + epsStep * xGradientCumulative.sign()
        xAdvStepOne = torch.min(xOriMax, xAdvStepOne)
        xAdvStepOne = torch.max(xOriMin, xAdvStepOne)
        xAdvStepOne = torch.clamp(xAdvStepOne.permute(0,2,3,1), clipMin, clipMax).permute(0,3,1,2)
        
        for m in range(len(tasks)):
            dFdX = dFdX + dFdXCompute(device, mtl_model, task, xAdvStepOne, yClean, criterionDict, normalize) 
        dXdAlpha = dXdAlphaCompute(device, fittingFactor, epsStep, alpha, dCdX, len(tasks), numSamples, xShape)
        
        dFdAlpha = torch.zeros(len(tasks), numSamples, xShape[0], xShape[1], xShape[2]).to(device)
        for m in range(len(tasks)):
            dFdAlpha[m] = dFdX * dXdAlpha[m]
        alpha = alpha - dFdAlpha * alphaLearningRate
        # print("alpha {}".format(alpha[0,0,0,50,50]))
        
        xGradientCumulativeTemp = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2]).to(device)
        for m in range(len(tasks)):
            xGradientCumulativeTemp = xGradientCumulativeTemp + alpha[m] * dCdX[m]
        xGradientCumulativeB = (decay * xGradientCumulativeB) + xGradientCumulativeTemp
        
        xAdv = xAdv + epsStep * xGradientCumulativeB.sign()
        xAdv = torch.min(xOriMax, xAdv)
        xAdv = torch.max(xOriMin, xAdv)
        xAdv = torch.clamp(xAdv.permute(0,2,3,1), clipMin, clipMax).permute(0,3,1,2)
        
        torch.cuda.empty_cache()
    return xAdv


def NativeGradient(device, mtl_model, task, xAdv, yClean, criterionDict, normalize=False):
    x = xAdv.detach().to(device)
    y = yClean[task].to(device)
    mask = yClean[task+'_mask'].to(device) if task+'_mask' in yClean else None
    
    x.requires_grad = True
    output = mtl_model(x, task=task)
    
    x.grad = None
    mtl_model.zero_grad()
    loss = criterionDict[task](output, y, mask)
    loss.mean().backward()
    xGrad = x.grad.detach()
    
    torch.cuda.empty_cache()
    return xGrad if not normalize else xGrad/loss.detach()[:,None,None,None]


def dFdXCompute(device, mtl_model, task, xAdvStepOne, yClean, criterionDict, normalize=False):
    return -NativeGradient(device, mtl_model, task, xAdvStepOne, yClean, criterionDict, normalize)
    

# Compute dX/dAlpha for each model m
def dXdAlphaCompute(device, fittingFactor, epsStep, alpha, dCdX, numTasks, numSamples, xShape, super = False):
    # Allocate memory for the solution
    dXdAlpha = torch.zeros(numTasks, numSamples, xShape[0], xShape[1], xShape[2]).to(device)
    innerSum = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2]).to(device)
    # First compute the inner summation sum m=1,...M: a_{m}*dC/dX_{m}
    for m in range(numTasks):
        innerSum = innerSum + alpha[m] * dCdX[m]
    # Multiply inner sum by the fitting factor to approximate the sign(.) function
    innerSum = innerSum * fittingFactor
    # Now compute the sech^2 of the inner sum
    innerSumSecSquare = SechSquared(innerSum)
    # Now do the final computation to get dX/dAlpha (may not actually need for loop)
    for m in range(numTasks):
        if super:
            dXdAlpha[m] = fittingFactor * epsStep[:,None][:,None][:,None] * dCdX[m] * innerSumSecSquare
        else:
            dXdAlpha[m] = fittingFactor * epsStep * dCdX[m] * innerSumSecSquare
    torch.cuda.empty_cache()
    return dXdAlpha

def SechSquared(x):
    y = 4 * torch.exp(2 * x) / ((torch.exp(2 * x) + 1) * (torch.exp(2 * x) + 1))
    return y