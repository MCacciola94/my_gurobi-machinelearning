import torch
import torch.nn as nn

#compute number of zero in weight masks of linear layers
def pruned_par(model):
    
    tot_pruned=0
    for m in model.modules():
        #Convolutional layers 
        if isinstance(m,torch.nn.Linear):
            if hasattr(m,'weight_mask'):
                el= float((m.weight_mask[:,:]==0).sum())
                tot_pruned+=el  

    return tot_pruned


#compute number of parmters of model, default takes into account all params from conv, linear and batchnorm
def par_count(model, bias=True):
    res = 0
    for m in model.modules():
        if isinstance(m,nn.Linear):
            res += par_count_module(m, bias=bias)

    return res

#count the number of a params of a module
def par_count_module(module, bias):
    res=0
    for name, par in module.named_parameters():
        if par.requires_grad and (bias or name!='bias'):
            res += par.numel()
    return res

#count the overall total number of a params of a model
def par_count_all(model):
    res=0
    for par in model.parameters():
        if par.requires_grad:
            res+= par.numel()
    return res


#compare two model printing different params
def find_diff_params(model1,model2, equal_vals= False):
    params1=model1.state_dict()
    params2=model2.state_dict()
    
    for par in params1.keys():
        if hasattr(par,'requires_grad') and par.requires_grad and par not in params2.keys():
            print('not in mod2 ', par)
        elif equal_vals and torch.is_tensor(params1[par]):
            if torch.norm(params1[par].to(torch.float)-params2[par].to(torch.float))!=0:
                print('different vals', par)


    for par in params2.keys():
        if hasattr(par,'requires_grad') and par.requires_grad and par not in params1.keys():
            print('not in mod1 ', par)
    