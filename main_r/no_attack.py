import torch
import numpy as np

def no_attack_results(device, model, tasks, valDataloader, criterionDict, metricDict):
    val_loss = {task: [] for task in tasks}
    for i, data in enumerate(valDataloader):
        xData = data['input'].to(device)
        # test acc. / loss
        with torch.no_grad():
            outputs = model(xData)
            for task in tasks:
                y = data[task].to(device)
                mask = data[task+'_mask'].to(device) if task+'_mask' in data else None
                metricDict[task](outputs[task], y, mask)
                val_loss[task].append(criterionDict[task](outputs[task], y, mask).mean().item())
                
    task_val_results = {}
    for task in tasks:
        val_results = metricDict[task].val_metrics()
        val_results['loss'] = np.mean(val_loss[task])
        task_val_results[task] = val_results
        print('Task {}: {}'.format(task[:4], val_results), flush=True)
    return task_val_results