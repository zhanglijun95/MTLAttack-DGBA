import pickle

def save_pickle(results, savePath):
    with open(savePath, 'wb') as handle:
        pickle.dump(results, handle)
            
def load_pickle(savePath):
    with open(savePath, 'rb') as handle:
        results = pickle.load(handle)
    return results

class clean_ARA():
    def __init__(self, data, backbone):
        if data=='NYUv2':
            self.tasks = ['segment_semantic','normal','depth_zbuffer']
            self.__metrics_prop = {'mIoU': False, 'Pixel Acc': False, 
                                  'Angle Mean': True, 'Angle Median': True, 'Angle 11.25': False, 'Angle 22.5': False, 'Angle 30': False,
                                  'abs_err': True,'rel_err': True,'sigma_1.25': False,'sigma_1.25^2': False,'sigma_1.25^3': False,
                                  'loss': True, 'acc.': False} # lower the better
            if 'resnet34' in backbone:
                self.ref_records = {'segment_semantic': {'mIoU': 0.2593, 'Pixel Acc': 0.5818, 'loss': 1.713534956886655}, 'normal': {'Angle Mean': 17.7864, 'Angle Median': 16.237, 'Angle 11.25': 29.127, 'Angle 22.5': 72.7137, 'Angle 30': 87.3004, 'loss': 0.06337280074755351}, 'depth_zbuffer': {'abs_err': 0.638, 'rel_err': 0.2472, 'sigma_1.25': 58.5251, 'sigma_1.25^2': 85.2873, 'sigma_1.25^3': 95.4123, 'loss': 0.6407556051299685}} # single-task models
        elif data=='Taskonomy':
            self.tasks = ['segment_semantic','normal','depth_zbuffer','keypoints2d','edge_texture']
            self.__metrics_prop = {'err': True, 'abs_err': True, 'cosine_similarity': False, 'key_err': True, 'edge_err': True, 'loss': True, 'acc.': False} # lower the better
            if 'resnet34' in backbone:
                self.ref_records = {'segment_semantic': {'err': 0.6199, 'loss': 0.5776133710993699}, 'normal': {'cosine_similarity': 0.8724, 'loss': 0.09638215413541039}, 'depth_zbuffer': {'abs_err': 0.022, 'loss': 0.021415408120334933}, 'keypoints2d': {'key_err': 0.2024, 'loss': 0.0863941674642374}, 'edge_texture': {'edge_err': 0.214, 'loss': 0.07870038644112974}} # single-task models
                
    def compute_rel(self, task_val_results, comp_results=None, verbose=True):
        if comp_results == None:
            comp_results = self.ref_records
        
        rel_results = {}
        for task in self.tasks:
            tmp = 0
            for metric in task_val_results[task]:
                value = task_val_results[task][metric]
                baseline = comp_results[task][metric]
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
        if verbose:
            print('Rel. Results: {}'.format(rel_results),flush=True)
            print('-'*100,flush=True)
        return rel_results
            