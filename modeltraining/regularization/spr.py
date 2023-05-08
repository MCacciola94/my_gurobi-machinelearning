import torch
import numpy as np

################################################################################
##################################### OPTIONS ##################################
################################################################################


class PerspReg: 
    def __init__(self, alpha, M, dim = 1, track_stats = False):
        self.alpha = alpha
        self.M = M
        self.dim = dim 
        self.const=(torch.sqrt(torch.Tensor([alpha/(1-alpha)])))
        self.track_stats=track_stats
        self.stats={'case1':0,'case2':0,'case3':0}

    #Computation of the current factor
    def __call__(self,net, lamb = 0.1):
        reg = 0    
        tot=0
        for m in net.modules():
            if isinstance(m,torch.nn.Linear):
                group = m.weight
                reg += self.compatible_group_computation(group, self.M[m])
                tot+=group.numel()

        reg = reg/tot

        return lamb* reg

    def compatible_group_computation(self, group, M):
        alpha=self.alpha
        const=self.const
        
        norminf=torch.norm(group,dim=self.dim ,p=np.inf)
        norm2= torch.norm(group,dim=self.dim ,p=2)
        num_el_struct=group.size(self.dim)
        


        bo1 = torch.max(norminf/M,const*norm2)>=1
        reg1 = alpha*norm2**2+1-alpha

        bo2 = norminf/M<=const*norm2
        reg2=2*(alpha*(1-alpha))**0.5*norm2

        eps=(torch.zeros(norminf.size()))
        eps=eps+1e-10
        reg3=alpha* norm2**2/(torch.max(eps,norminf))*M+(1-alpha)*norminf/M

        bo2=torch.logical_and(bo2, torch.logical_not(bo1))
        bo3=torch.logical_and(torch.logical_not(bo2), torch.logical_not(bo1))

        reg=(bo1*reg1+bo2*reg2+bo3*reg3).sum()*num_el_struct
        if self.track_stats:
            self.stats['case1']+=bo1.sum().item()
            self.stats['case2']+=bo2.sum().item()
            self.stats['case3']+=bo3.sum().item()
                        
        return reg

   