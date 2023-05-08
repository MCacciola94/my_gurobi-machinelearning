#!/usr/bin/env python
# coding: utf-8
"""
Train or load a model on the MNIST dataset
Optionally solve the adversarial problem defined by it
"""

import os
from time import time
import random

import numpy as np
import pandas as pd
import torch

from modeltraining import architecture, trainer
import utils
from mipoptimization.mnistadversarial import adversarial
from modeltraining import pruningutils as pu


def run(config):
    save_path = utils.getSavePath(config)
    if os.path.isfile(save_path) and not(config.dont_save): # exist res
        df = pd.read_csv(save_path)
        skip = True # skip flag
    else:
        df = pd.DataFrame(columns=["Acc", "Loss", "Elapsed", "Epochs"])
        skip = False # skip flag

    for i in range(config.expnum):
        # random seed for each experiment
        config.seed = i
        # set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        # start exp
        print("===============================================================")
        print("Experiment {}:".format(i), flush =True)
        print("===============================================================")
        # generate data
        print()
        # skip exist experiments
        if skip and (i < len(df)):
            print("Skip experiment {}.".format(i))
            print()
            continue
        else:
            skip = False


        # Call of the algorithm
        if config.pretrained_path == '':                
            # load architecture
            model = architecture.load_arch(config.arch)
            print()
            #train the model
            acc, loss = trainer.train(model, config)
        elif os.path.isfile(config.pretrained_path):
                # load checkpoint
                model, acc, loss = utils.load_chkpt(config.pretrained_path)
        else:
            print("=> no checkpoint found at '{}'".format(config.pretrained_path))
            return 0
        
        # if no adversarial part is needed elapsed is -1
        elapsed = -1 
        if not config.trainonly:
            # reformat the name of the model parms so it works with the gurobi pkg (if needed)
            pu.rm_add_par(model)
            tick = time()
            # adversarial part of the experiment
            adversarial(model, config)
            tock = time()
            elapsed = tock - tick

        # printing and saving
        print("Time elapsed: {:.4f} sec".format(elapsed))
        print()
        # save

        row = {"Acc":acc,"Loss":loss.item(),
               "Elapsed":elapsed, "Epochs":config.epochs}
        df = df.append(row, ignore_index=True)
        if not config.dont_save:
            df.to_csv(save_path, index=False)
            print("Saved to " + save_path + ".")
        else:
            print("DONT_SAVE OPTION ACTIVE, RESULTS NOT SAVED")

        print("\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # general configuration
    parser.add_argument("--trainonly",
        action="store_true",
        help="only train the ML model")
    parser.add_argument("--prune",
        action="store_true",
        help="prune the model")
    parser.add_argument("--expnum",
                    type=int,
                    default=1,
                    help="number of experiments")
    parser.add_argument("--dont_save",
        action="store_true",
        help="save traied model")
    
    # model configuration
    parser.add_argument("--arch",
                        type=str,
                        default="2x50",
                        help="architecture of the model")

    # paths configurations
    parser.add_argument("--pretrained_path",
                        type=str,
                        default="",
                        help="path to pretrained model")
    parser.add_argument("--path",
                        type=str,
                        default="./res/last",
                        help="path to save result")
    parser.add_argument("--save_path",
                    type=str,
                    default="./saved_models",
                    help="path to save models")

    # Training configuration
    parser.add_argument("--epochs",
                        type=int,
                        default=2,
                        help="number of epochs")
    parser.add_argument("--optim",
                        type=str,
                        default="sgd",
                        choices=["sgd", "adam"],
                        help="optimizer neural network")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-1,
                        help="learning rate")
    parser.add_argument("--wd",
                    type=float,
                    default=0.0,
                    help="weight decay")
    parser.add_argument("--momentum",
                type=float,
                default=0.0,
                help="momentum")
    parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="batch size")
    parser.add_argument("--print_freq",
                    type=int,
                    default=100,
                    help="print frequence for training")
    # regularization parameters
    parser.add_argument("--reg",
                        type=str,
                        default="spr",
                        choices=["spr", "l1l2", "l1linf"],
                        help="regularization for pruning")
    parser.add_argument("--lamb",
                    type=float,
                    default=0.0,
                    help="lambda for regularization")
    parser.add_argument("--alpha",
                    type=float,
                    default=0.0,
                    help="alpha spr param")
    parser.add_argument("--M",
                        type=str,
                        default="layer",
                        choices=["layer","param"],
                        help="Big M setting")
    parser.add_argument("--dim",
                    type=int,
                    default=1,
                    choices=[0,1],
                    help="prune on what dimension")
    
    # pruning parameters
    parser.add_argument("--ft_epochs",
                        type=int,
                        default=1,
                        help="epochs after pruning")
    parser.add_argument("--threshold",
                    type=float,
                    default=5e-2,
                    help="accuracy drop threshold for pruning")   
    parser.add_argument("--threshold_str",
                    type=float,
                    default=5e-3,
                    help="accuracy drop threshold for pruning")  
    # adversarial problem configuration
    parser.add_argument("--samp_id",
            type=int,
            default=10000,
            help="id of the sample to use for adversarial problem")
    parser.add_argument("--time",
                    type=float,
                    default=180,
                    help="mip time limit")
    
    # get configuration
    config = parser.parse_args()
    # run experiment pipeline
    run(config)

