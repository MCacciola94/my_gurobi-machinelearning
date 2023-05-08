from torch.nn.utils import prune
import torch
import torch.nn as nn

from modeltraining import pruningutils as pu
import os

def load_arch(arch, pretrained = "", already_pruned = True):

    model = build_model(arch)
    

    if not(pretrained == "")  and already_pruned:
        for m in model.modules(): 
            if hasattr(m, 'weight'):
                pruning_par=[((m,'weight'))]

                if hasattr(m, 'bias') and not(m.bias==None):
                    pruning_par.append((m,'bias'))

                prune.global_unstructured(pruning_par, pruning_method=pu.ThresholdPruning, threshold=1e-18)

                
    # optionally resume from a checkpoint

    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading pretrained model from '{}'".format(pretrained))
            checkpoint = torch.load(pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pretrained with accuracy '{}' (epochs {})"
                .format(checkpoint['best_prec1'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

    return model

def build_model(arch):
    """
    multi-layer fully connected neural network regression
    """
    arch = arch.split('-')
    arch = [elem.split('x') for elem in arch]
    aux =[]
    for el in arch:
        n, size = int(el[0]), int(el[1])
        aux += [size for _ in range(n)]

    arch = aux

    arch = [28*28] + arch + [10]
    layers = []
    for i in range(len(arch)-1):
        layers.append(nn.Linear(arch[i], arch[i+1]))
        if i < len(arch) - 2:
            layers.append(nn.ReLU())
            
    return nn.Sequential(*layers)



