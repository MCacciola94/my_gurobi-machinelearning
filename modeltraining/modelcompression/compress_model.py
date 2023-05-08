import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import copy


# return the indexes of filters that are not completely pruned maskwise
def get_unpruned_neurons(m):
    idx = []
    if isinstance(m,nn.Linear):
        for i in range(m.out_features):
            if m.weight_mask[i].sum()>0:
                idx.append(i)
    
    return idx

#remove parameters created by pytorch pruning function in linear layers
def remove_additional_pars_linear(linear):

    weight_bkp = linear.weight.data
    prune.remove(linear,'weight')
    linear.weight=nn.Parameter(weight_bkp)
    
    if hasattr(linear,'bias') and linear.bias is not None:
        bias_bkp = linear.bias.data
        prune.remove(linear,'bias')
        linear.bias=nn.Parameter(bias_bkp)

# compress a linear layer selecting the desired input features
def compress_linear_in(lin, idx_not_pr):
    weight = lin.weight.data
    weight = torch.index_select(weight, dim = 1, index = idx_not_pr)
    lin.weight = nn.Parameter(weight)
    lin.in_features = lin.weight.shape[1]

# compress a linear layer selecting the desired input features
def compress_linear_out(lin, idx_not_pr):
    weight = lin.weight.data
    weight = torch.index_select(weight, dim = 0, index = idx_not_pr)
    bias = lin.bias.data
    bias = torch.index_select(bias, dim = 0, index = idx_not_pr)
    lin.weight = nn.Parameter(weight)
    lin.bias = nn.Parameter(bias)
    lin.out_features = lin.weight.shape[0]

# Copress an isolated sequence conv1->batchnorm->conv2, where the compression is consequence of the maskwise pruned filters of conv1
def compress_lin_relu_lin(lin1,lin2):
    
    idx_not_pr = get_unpruned_neurons(lin1)
    
    if len(idx_not_pr) < lin1.out_features:
        if len(idx_not_pr) == 0:
            print("seems to not be connected")
            idx_not_pr = [0]
            
        idx_not_pr = torch.Tensor(idx_not_pr).type(torch.int)
        compress_linear_out(lin1,idx_not_pr)
        compress_linear_in(lin2,idx_not_pr)

    return idx_not_pr

#Compress a resnet model that already have a pruning mask
def compress_sequential(net):
# Compress block by block, info about the next block is necessary
    for i, layer in enumerate(net):
        if i%2==0 and i+2< len(net):
            compress_lin_relu_lin(layer, net[i+2])

# Remove masks and other params
    for module in net.modules():
        if isinstance(module,nn.Linear):
            remove_additional_pars_linear(module)


from modeltraining.architecture import load_arch
from modeltraining import pruningutils as pu
from modeltraining import trainer
from modeltraining.modelcompression import comprutils

def unit_test_seq(name):

    base_checkpoint=torch.load(name)
    arch =base_checkpoint['arch']
    model_pruned= load_arch(arch)

    pu.prune_thr(model_pruned,1.e-30)


    model_pruned.load_state_dict(base_checkpoint['state_dict'])

    model_pruned.eval()
    model_pruned(torch.rand([1,28*28]))

    dataset = trainer.get_data()
    criterion = nn.CrossEntropyLoss()
    model_pruned(torch.rand([1,28*28]))

    optimizer_pruned = torch.optim.SGD(model_pruned.parameters(), 0.0,
                                momentum=0.0,
                                weight_decay=0)


    trainer.validate(dataset, model_pruned)

    # breakpoint()
    model_aux = copy.deepcopy(model_pruned)
 

    _, sp_rate0_pr = pu.sparsityRate(model_pruned)
    pr_par0_pr = comprutils.pruned_par(model_pruned)
    tot0_pr = comprutils.par_count(model_pruned)

    compress_sequential(model_pruned)



    #print(model)
    # new_tot = par_count(model)
    # trainer.validate(reg_on = False)
    trainer.validate(dataset, model_pruned)
    



    _, sp_rate1_pr = pu.sparsityRate(model_pruned)
    pr_par1_pr = comprutils.pruned_par(model_pruned)
    tot1_pr =comprutils.par_count(model_pruned)
    # print("tot berfore and after ",tot0, ' ', tot1, ' pr ', tot0_pr,' ', tot1_pr)
    #print("new_tot ", new_tot)
    # print("sparsity before and after ", sp_rate0, ' ', sp_rate1, ' pr ', sp_rate0_pr,' ', sp_rate1_pr )
    # print("pruned par before and after ", pr_par0, ' ', pr_par1, ' pr ', pr_par0_pr,' ', pr_par1_pr )

    print('New stats ', tot0_pr-tot1_pr,' perc ', 100*(tot0_pr-tot1_pr)/tot0_pr )
    # fully_pruned= resnet_pruned.FullyCompressedResNet(copy.deepcopy(model_pruned))
    # breakpoint()


    # trainer_fully_pruned = Trainer(model = fully_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
    #                                         criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)
    # print('Fully pruned model acc ')
    # trainer_fully_pruned.validate(reg_on = False)


 

    # _, sp_rate_fll = at.sparsityRate(fully_pruned)
    # pr_par_fll = pruned_par(fully_pruned)
    # tot_fll = par_count(fully_pruned)

    # print('Final stats ', tot0_pr-tot_fll,' perc ', 100*(tot0_pr-tot_fll)/tot0_pr )


    return model_pruned, dataset

