import torch
from torch import nn
import numpy as np

def get_M_dict(model, config, const = False, scale =1.0):
     if config.M == 'layer':
          return layerwise_M(model, const = False, scale = 1.0)
     if config.M == 'param':
          return paramwise_M(model, const = False, scale = 1.0)
     
     return None
     
def layerwise_M(model, const = False, scale = 1.0):
       Mdict={}
       if const:
           for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=1.0
       else:
            for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=torch.norm(m.weight,p=np.inf).item() * scale

       return Mdict

def paramwise_M(model, const = False, scale = 1.0):
    Mdict={}
    model.eval()
    with torch.no_grad:
        if const:
            for par in model.parameters():
                    if par.requires_grad:
                        Mdict[par]=scale
        else:
                for par in model.parameters():
                    if par.requires_grad:
                        Mdict[par]=torch.norm(par.data,p=np.inf).item() * scale

        return Mdict
