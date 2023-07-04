from torch.nn.utils import prune
import torch
import torch.nn as nn
import numpy as np


#Pruning crieterion for unstructured threshold pruning
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold
    
def prune_thr(model, thr):
    for m in model.modules(): 
      if isinstance(m,torch.nn.Linear):
          pruning_par=[((m,'weight'))]
  
          if hasattr(m, 'bias') and not(m.bias==None):
              pruning_par.append((m,'bias'))
  
          prune.global_unstructured(pruning_par, pruning_method=ThresholdPruning, threshold=thr)

def prune_struct(model, thr = 0.05):
    for m in model.modules():
        if isinstance(m,torch.nn.Linear):
            for i in range(m.out_features):
                if m.weight_mask[i,:].sum()/m.weight_mask[i,:].numel()>thr:
                    m.weight_mask[i,:]=1
                else:
                    m.weight_mask[i,:]=0

def param_saving(layers, skip = 1 , freq = 2, filter_size = 9):
    layers = layers[1:]
    first = 0
    second = 1
    tot = 0
    while second < len(layers):
        pruned_filetrs = sum([int(e) for e in layers[first]])
        rem_filters = len(layers[second]) - sum([int(e) for e in layers[second]])
        print(first, second, pruned_filetrs, rem_filters )
        tot += filter_size * pruned_filetrs * rem_filters
        first += freq
        second += freq
    return tot

def retrive_arch(model):
    if not isinstance(model, torch.nn.Sequential):
        print('Not a sequential model')
        return None
    arch = []
    skip = False
    for layer in model[2:]:
        if not skip:
            if not isinstance(layer, torch.nn.Linear):
                print("This is not a linear layer!")
                return None
            arch.append(layer.in_features)
            skip = True
        else: skip = False

    arch_str = ''
    size = arch[0]
    n = 0 
    for el in arch:
        if el == size:
            n+=1
        else:
            arch_str += '-'+str(n)+'x'+str(size)
            size = el
            n = 1
    arch_str += '-'+str(n)+'x'+str(el)

    return arch_str[1:]


#method that prune neurons of the model based on their sparsity
def thresholdNeuronPruning(module,mask,threshold=0.95):

    #Fully Connected layers
   if isinstance(module,nn.Linear):
        for i in range(module.out_features):
            if float((module.weight[i,:]==0).sum()/module.weight[i,:].numel())>threshold:
                mask[i]=0

    #Convolutional layers    
   if isinstance(module,nn.Conv2d):
        for i in range(module.out_channels):
            if float((module.weight[i,:,:,:]==0).sum()/module.weight[i,:,:,:].numel())>threshold:
                mask[i]=0
            
   return 1-mask.sum()/mask.numel()    


#Computing sparsity information of the model
def sparsityRate(model,verb_lev=-1,opt="channels"):
    # breakpoint()

    #retrocompatible with previous version
    if verb_lev==False:
        verb_lev=0
    elif verb_lev== True:
        verb_lev=1

    out=[]
    tot_pruned=0
    tot_struct_pruned=0
    for m in model.modules():
        #Fully Connected layers
        if isinstance(m,nn.Linear):
            v=[]
            for i in range(m.out_features):
                el=float((m.weight[i,:]==0).sum()/m.weight[i,:].numel())
                v=v+[el]
                tot_pruned+=m.in_features*el
                if el==1.0:
                    tot_struct_pruned+=m.in_features
                    
                
            if verb_lev==1:
                print("in module ",m,"\n sparsity of  ", v)
            elif verb_lev==0:
                print("\n sparsity of  ", v)
            out=out+[v]

    return out,(tot_pruned,tot_struct_pruned)


        


# compute the maximum weight for each neuron
def maxVal(model): 
    out=[]
    for m in model.modules():
          #Fully Connected layers
        if isinstance(m,nn.Linear):
            # for i in range(m.out_features):  
            #     v=v+ [float(torch.norm(m.weight[i,:],np.inf))]
            v=torch.norm(m.weight,dim=1,p=np.inf)
        else:
            continue

        print("\nmax weight is ", v)
        out+=[v]
    return out

#remove parameters created by pytorch pruning function in a model
# the pruning is now permanent and saved in the original params
def rm_add_par(model):
    for m in model.modules():
        if hasattr(m,'weight'):
            if not('weight' in dict(m.named_parameters()).keys()):
                prune.remove(m,'weight')
                if hasattr(m,'bias') and m.bias is not None:
                    if not('bias' in dict(m.named_parameters()).keys()):
                        prune.remove(m,'bias')
