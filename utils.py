import os
import torch
from modeltraining import pruningutils as pu
from modeltraining import architecture as archs


def getSavePath(config):
    """
    get file path to save result
    """
    path = config.path
    if not os.path.isdir(path):
        os.mkdir(path)

    if config.pretrained_path != '':
        res_path = config.pretrained_path.split('saved_models')[-1]
        res_path = res_path.replace('/','|')
        filename = "RESUME_" + res_path
    else:
        # arch and epochs

        filename = "ARCH_{}-EPOCHS_{}".format(config.arch, config.epochs)

        # if different from defoult
        opt = '-OPT_' + config.optim if config.optim != 'sgd' else ''
        lr = '-LR_' + str(config.lr) if config.lr != 1e-1 else ''
        wd = '-WD_' + str(config.wd) if config.wd != 0.0 else ''
        mom = '-MOM_' + str(config.momentum) if config.momentum != 0.0 else ''
        bs = '-BS_' + str(config.batch_size) if config.batch_size != 128 else ''

        # pruning info
        reg = '-REG_' + config.reg if config.prune else ''
        lamb = '-LAMB_' + str(config.lamb) if config.prune else ''
        alpha = '-ALPHA_' + str(config.alpha) if config.prune else ''
        fte = '-FT_' + str(config.ft_epochs) if config.prune else ''

        # additional infos (only if diff from def)
        thr = '-THR_' + str(config.threshold) if config.prune and config.threshold !=  5e-2 else ''
        thr_str = '-THRSTR_' + str(config.threshold_str) if config.prune and config.threshold_str !=  5e-3 else ''
        dim = '-DIM_' + str(config.dim) if config.prune and config.dim !=  1 else ''

        filename += opt + lr + wd + mom + bs + reg + lamb + alpha + fte + thr + thr_str + dim 


    # adversarial part info
    id = '-ID_' + str(config.samp_id) if not config.trainonly else ''
    time = '-TIME_' + str(config.time) if not(config.trainonly) and config.time != 180 else ''
    
    filename += id + time

    if not(config.dont_save) and config.pretrained_path == '':
        if not os.path.isdir(config.save_path):
            os.mkdir(config.save_path)
        config.save_path = config.save_path + "/" + filename 
        if not os.path.isdir(config.save_path):
            os.mkdir(config.save_path)
    
    print(filename)

    return path + "/" + filename + ".csv"

def load_chkpt(path):
    print("=> loading pretrained model from '{}'".format(path))
    checkpoint = torch.load(path)
    arch = checkpoint['arch']
    model = archs.load_arch(arch)

    if "checkpointPR_" in path.split('/')[-1]:
        pu.prune_thr(model, thr=1e-30)
        model(torch.rand(1,28*28))

    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded pretrained arch {}, with accuracy '{}'"
        .format(checkpoint['best_prec1'],arch))
    return model, checkpoint['best_prec1'], checkpoint['loss']