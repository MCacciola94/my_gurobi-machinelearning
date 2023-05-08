from modeltraining.regularization import spr
from modeltraining.regularization import classicreg as creg
from modeltraining.regularization import regutils
import torch
from random import random
import os

def get_fun(model, config):
    name= config.reg
    if 'spr' in name:
        # temporary backup  of current weights
        rand_id = random()
        torch.save(model.state_dict(), str(rand_id) + "_rand_init.ph")

        # load bigM values sotred in a model
        if os.path.isfile("./saved_models/bigMs/ARCH_" + config.arch +  "/checkpoint.th"):
            base_checkpoint=torch.load("./saved_models/bigMs/ARCH_" + config.arch +  "/checkpoint.th")
            model.load_state_dict(base_checkpoint['state_dict'])

            # store big M in the approriate structure
            M = regutils.get_M_dict(model, config)

            # reload starting values
            model.load_state_dict(torch.load(str(rand_id) + "_rand_init.ph"))
            os.remove(str(rand_id) + "_rand_init.ph")

        else: #uses current values
            # store big M in the approriate structure
            M = regutils.get_M_dict(model, config)

        return spr.PerspReg(config.alpha, M)
    else:
        return creg.__dict__[name](config.alpha)
