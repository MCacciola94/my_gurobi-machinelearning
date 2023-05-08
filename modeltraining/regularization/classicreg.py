import torch
import numpy as np

################################################################################
##################################### OPTIONS ##################################
################################################################################

SCALED = True
class l1l2: 
    def __init__(self,alpha, dim = 1):
        self.alpha = alpha
        self.dim =dim
    #Computation of the current factor
    def __call__(self, net, lamb = 0.1):

        reg = 0    
        tot=0
        for m in net.modules():
            if isinstance(m,torch.nn.Linear):
                group = torch.cat((m.weight,m.bias.unsqueeze(1)),dim=1)
                reg += self.compatible_group_computation(group)
                tot += group.numel()

        if not SCALED:
            tot=1

        reg = reg/tot


        return lamb* reg

    def compatible_group_computation(self, group):
        alpha= self.alpha
        norm2= torch.norm(group, dim=self.dim, p=2)
        if SCALED:
            num_el_struct=group.size(self.dim)
        else: num_el_struct =1

        reg=(alpha * (norm2**2).sum() + ((1-alpha) * norm2).sum() )*num_el_struct
                        
        return reg



    
    
class perspregcasex: 
    def __init__(self,alpha,M,case, option= 'convs_and_batchnorm',  ):
        self.alpha=alpha
        self.M=M
        self.const=(torch.sqrt(torch.Tensor([alpha/(1-alpha)]))).cuda()
        self.option=option
        self.case =case
    #Computation of the current factor
    def __call__(self,net, lamb = 0.1):

        if self.option=='single_convs':
            reg= self.single_convs(net)

        if self.option=='convs_and_batchnorm':
            reg= self.convs_and_batchnorm(net)

        return lamb* reg

    def compatible_group_computation(self,group, M):
        alpha=self.alpha
        const=self.const
        
        norminf=torch.norm(group,dim=1,p=np.inf)
        norm2= torch.norm(group,dim=1,p=2)
        num_el_struct=group.size(1)
        


        # bo1 = torch.max(norminf/M,const*norm2)>=1
        if self.case ==1: 
            reg = alpha*norm2**2+1-alpha
        elif self.case == 2:

        # bo2 = norminf/M<=const*norm2
            reg=2*(alpha*(1-alpha))**0.5*norm2
        elif self.case ==3:
            eps=(torch.zeros(norminf.size())).cuda()
            eps=eps+1e-10
            reg=alpha* norm2**2/(torch.max(eps,norminf))*M+(1-alpha)*norminf/M
        

        # bo2=torch.logical_and(bo2, torch.logical_not(bo1))
        # bo3=torch.logical_and(torch.logical_not(bo2), torch.logical_not(bo1))

        reg=reg.sum()*num_el_struct
                        
        return reg

    def single_convs(self,net):
        reg = 0    
        tot=0
        for m in net.modules():
            if isinstance(m,torch.nn.Conv2d):
                group = m.weight
                reg+=self.compatible_group_computation(group.reshape(group.size(0),-1), self.M[m])
                tot+=group.numel()

        return reg/tot

    #mode is supposed to  have a method to retrive conv-batchnorm blocks
    def convs_and_batchnorm(self,net):
        if isinstance(net,torch.nn.DataParallel):
            net=net.module
        reg = 0    
        tot=0
        for block in net.conv_batchnorm_blocks():
            conv = block['conv']
            bnorm = block['batchnorm']
            conv_w = conv.weight
            conv_w=conv_w.reshape(conv_w.size(0),-1)
            group = torch.cat((conv_w,bnorm.weight.unsqueeze(1),bnorm.bias.unsqueeze(1)),dim=1)
            if conv.bias is not None:
                group = torch.cat((group,conv.bias.unsqueeze(1)),dim=1)

            reg+=self.compatible_group_computation(group,self.M[block['id']])
            tot+=group.numel()

        return reg/tot


def perspregcase1(alpha,M, option= 'convs_and_batchnorm'): 
    return perspregcasex(alpha,M, case=1, option=option)

def perspregcase2(alpha,M, option= 'convs_and_batchnorm'): 
    return perspregcasex(alpha,M, case=2, option=option)

def perspregcase3(alpha,M, option= 'convs_and_batchnorm'): 
    return perspregcasex(alpha,M, case=3, option=option)
  
    
