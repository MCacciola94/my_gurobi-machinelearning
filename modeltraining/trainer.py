import os
from time import time
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from modeltraining import pruningutils as pu
from modeltraining.modelcompression.compress_model import compress_sequential

from modeltraining.regularization import regterm


BN_SRCH_ITER = 6

# get optimizer
def get_optimizer(model, config):
    lr = config.lr
    momentum = config.momentum
    wd = config.wd
    if config.optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr, weight_decay=wd, momentum=momentum)
    elif config.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr, weight_decay=wd, momentum=momentum)
    else:
        print('Invalid optimizer')
        return None
    
def get_data(config = None):
    if config is None:
        batch_size =128
    else:
        batch_size=config.batch_size
    if config is None or config.dataset == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root="/local1/caccmatt/Pruning_for_MIP/Dataset", train=True,
                                            transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root="/local1/caccmatt/Pruning_for_MIP/Dataset", train=False,
                                                transform=transforms.ToTensor(), download=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                                                shuffle=True,num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,
                                                shuffle=False,num_workers=8, pin_memory=True)
    elif config.dataset == 'Cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset = torchvision.datasets.CIFAR10(root="/local1/caccmatt/Pruning_for_MIP/Dataset", train=True,
                                            transform=transforms.Compose([
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomCrop(32, 4),
                                                                    transforms.ToTensor(),
                                                                    normalize,
                                                                ]), download=True)
        test_dataset = torchvision.datasets.CIFAR10(root="/local1/caccmatt/Pruning_for_MIP/Dataset", train=False,
                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    normalize,
                                                                ]), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                                                shuffle=True,num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,
                                                shuffle=False,num_workers=4, pin_memory=True)
    else:
        print('Dataset not available, aborting')
        return 0
    
    return {'train_loader': train_loader, "valid_loader": test_loader}

def run_epoch(epoch, dataset, model,optimizer, criterion, config, reg_on = False, reg=None):
    """
        Run one training epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time()
    for i, (input, target) in enumerate(dataset["train_loader"]):

        input = input.reshape(-1,28*28)
        # compute output
        output = model(input)

        loss = criterion(output, target)

        loss_noreg = loss.item()
        if reg_on:
            regTerm_gd = reg(model, config.lamb)
            regTerm = regTerm_gd.item()
            loss += regTerm_gd
        else: regTerm = 0.0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.detach().float()
        loss = loss.detach().float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))

        # measure elapsed time
        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f}) ([{lnrg:.3f}]+[{lrg:.3f}])\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(dataset["train_loader"]), loss = losses, lnrg = loss_noreg, lrg = regTerm, top1 = top1))
        
    
    return loss

def validate(dataset, model, config = None, reg_on = False, reg = None):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    print_freq = 100 if config is None else config.print_freq

    criterion = torch.nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()
    end = time()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataset["valid_loader"]):
            
            input = input.reshape(-1,28*28)
            # compute output
            output = model(input)
            loss = criterion(output, target)
            loss_noreg = loss.item()

            if reg_on:
                regTerm_gd=reg(model, config.lamb)
                regTerm = regTerm_gd.item()
                loss+=regTerm_gd
            else: regTerm = 0.0

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))
            
            if (i+1) % (print_freq//2) == 0:
                print('Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) ([{lnrg:.3f}]+[{lrg:.3f}])\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(dataset["valid_loader"]), loss = losses, lnrg = loss_noreg, lrg = regTerm,
                            top1 = top1))
    print(' * Prec@1 {top1.avg:.3f}'
            .format(top1 = top1))


    return top1.avg
   
def binary_thr_search(model, dataset, iters, config):
    model.eval()

    print('UNSTRUCT THRESHOLDING')

    a=0
    b=1e-1
    last_feas_thr=a
    
    valid_loader_bck = dataset["valid_loader"]
    print_freq_bkp =config.print_freq
    print_freq= int(1e10)
    dataset["valid_loader"]=dataset["train_loader"]
    print('original accuracy')
    original_acc = validate(dataset, model, config, reg_on = False)

    pu.prune_thr(model,1e-30)
    validate(dataset, model, config, reg_on = False)

    original_state = model.state_dict()
    
    for i in range(iters):
        thr=a+(b-a)*0.5
        print('current threshold ', thr)
        pu.prune_thr(model,thr)
        acc = validate(dataset, model, config, reg_on = False)
        
        if original_acc-acc<config.threshold:
            a=thr
            last_feas_thr=thr
        else :
            b=thr
            model.load_state_dict(original_state)
            model(torch.rand([1,28*28]))
            # print('resuming')
            # validate(dataset, model, config, reg_on = False)

    model.load_state_dict(original_state)
    model(torch.rand([1, 28*28]))

    pu.prune_thr(model,last_feas_thr)
    print('Final unstruct threshold ', last_feas_thr)
    validate(dataset, model, config, reg_on = False)

    dataset["valid_loader"]=valid_loader_bck
    config.print_freq= print_freq_bkp
 
def binary_thr_struct_search(model, dataset, iters, config):
    model.eval()
    print('STRUCT THRESHOLDING')

    a=0
    b=1e-1
    last_feas_thr=a
    
    valid_loader_bck = dataset["valid_loader"]
    print_freq_bkp =config.print_freq
    config.print_freq= int(1e10)
    dataset["valid_loader"]=dataset["stable_train_loader"]
    print('original accuracy')
    original_acc = validate(dataset, model, config, reg_on = False)
    
    pu.prune_thr(model,1e-30)
    validate(dataset, model, config, reg_on = False)

    original_state = model.state_dict()
    
    for i in range(iters):
        thr=a+(b-a)*0.5
        print('current threshold ', thr)

        pu.prune_struct(model,thr)
        
        acc = validate(dataset, model, config, reg_on = False)

        if original_acc-acc<config.threshold_str:
            a=thr
            last_feas_thr=thr
        else:
            b=thr
            model.load_state_dict(original_state)
            model(torch.rand([1,28*28]))
            print('resuming')
            validate(dataset, model, config, reg_on = False)

    model.load_state_dict(original_state)
    model(torch.rand([1,28*28]))

    pu.prune_struct(model,last_feas_thr)
    print('Final struct threshold ', last_feas_thr)
    validate(dataset, model, config, reg_on = False)



    dataset["valid_loader"]=valid_loader_bck
    config.print_freq= print_freq_bkp

def train(model, config):
    epochs = config.epochs
    dataset =get_data(config)
    optimizer = get_optimizer(model, config)   
    criterion = torch.nn.CrossEntropyLoss()
    if config.prune:
        reg = regterm.get_fun(model, config)
    else: reg = None 
    
    # Starting time
    start = time()        

    for epoch in range(epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        
        loss = run_epoch(epoch, dataset, model, optimizer, criterion, config, reg_on=config.prune, reg = reg)
        # evaluate on validation set
        prec1 = validate(dataset, model, config, reg_on = config.prune, reg=reg)

    if not config.dont_save:
        filename = os.path.join(config.save_path, 'checkpoint_0.th')
        i=1
        while os.path.isfile(filename):
            filename = os.path.join(config.save_path, 'checkpoint_'+str(i)+'.th')
            i+=1
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': prec1,
            'arch': config.arch,
            'loss': loss
        }, filename=filename)

    elapsed = time()-start
    print("\n Elapsed time for training ",elapsed)

    if not config.prune:
        return prec1, loss

    spars, tot_p = pu.sparsityRate(model)
    print("Total parameter pruned without thresholding:", tot_p[0], "(unstructured)", tot_p[1], "(structured)")

    # pu.maxVal(self.model)   
    # Pruning parameters under the threshold

    binary_thr_search(model, dataset, BN_SRCH_ITER, config)
    spars, tot_p = pu.sparsityRate(model)
    
    print("\n Total parameter pruned after unstruct thresholding:", tot_p[0], "(unstructured)", tot_p[1],"(structured)\n")

    validate(dataset, model,config, reg_on = True, reg=reg)

    #recovering all pruned weights that are not in a pruned entity
    # self.binary_thr_struct_search(10)
    pu.prune_struct(model, config.threshold_str)

    spars, tot_ = pu.sparsityRate(model)
    

    print("\n Total parameter pruned after struct thresholding:", tot_p[0], "(unstructured)", tot_p[1],"(structured)\n")

    validate(dataset, model,config, reg_on = True, reg=reg)
    #Finetuning of the pruned model
    print("\n Total elapsed time ", time()-start,"\n FINETUNING\n")
    best_prec1 = 0
    id_best = -1

    for epoch in range(epochs, epochs + config.ft_epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss = run_epoch(epoch, dataset, model, optimizer, criterion, config, reg_on = False)

        # evaluate on validation set
        prec1 = validate(dataset, model,config, reg_on = False)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best: loss_tr = loss

        if is_best:
            best_pars =model.state_dict()
            if not config.dont_save:
                if id_best == -1:
                    filename = os.path.join(config.save_path, 'checkpointPR_0.th')
                    i=0
                    while os.path.isfile(filename):
                        i+=1
                        filename = os.path.join(config.save_path, 'checkpointPR_'+str(i)+'.th')
                        
                    id_best = i
                else: filename = os.path.join(config.save_path, 'checkpointPR_'+str(id_best)+'.th')
                save_checkpoint({
                    'state_dict': best_pars,
                    'best_prec1': best_prec1,
                    'arch': config.arch,
                    'loss': loss_tr
                }, filename=filename)

    elapsed = time()-start
    print("\n Elapsed time for training ", elapsed)

    model.load_state_dict(best_pars)
    spars, tot_p = pu.sparsityRate(model)

    print("Total parameter pruned:", tot_p[0], "(unstructured)", tot_p[1],"(structured)")
    
    validate(dataset, model,config, reg_on = False)
    print("Best accuracy: ", best_prec1)

    compress_sequential(model)
    pruned_arch = pu.retrive_arch(model)

    if not config.dont_save:
        filename = os.path.join(config.save_path, 'checkpointRED_0.th')
        i=1
        while os.path.isfile(filename):
            filename = os.path.join(config.save_path, 'checkpointRED_'+str(i)+'.th')
            i+=1

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'arch': pruned_arch,
            'loss': loss_tr
        }, filename=filename)
    validate(dataset, model,config, reg_on = False)
    
    return best_prec1, loss_tr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


