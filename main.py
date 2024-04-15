import os
import argparse
import itertools


from experiment import run

# set parser
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
    help="save trained model")
# dataset
parser.add_argument("--dataset",
                type=str,
                default="MNIST",
                help="dataset for training")
    # model configuration
# # model configuration
# parser.add_argument("--arch",
#                     type=str,
#                     default="2x50",
#                     help="architecture of the model")

# paths configurations
parser.add_argument("--pretrained_path",
                    type=str,
                    default="",
                    help="path to pretrained model")
parser.add_argument("--path",
                    type=str,
                    default="./res/GRID/last",
                    help="path to save result")
parser.add_argument("--save_path",
                type=str,
                default="./saved_models",
                help="path to save models")

# Training configuration
parser.add_argument("--epochs",
                    type=int,
                    default=50,
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
# parser.add_argument("--reg",
#                     type=str,
#                     default="spr",
#                     choices=["spr", "l1l2", "l1linf"],
#                     help="regularization for pruning")
# parser.add_argument("--lamb",
#                 type=float,
#                 default=0.0,
#                 help="lambda for regularization")
# parser.add_argument("--alpha",
#                 type=float,
#                 default=0.0,
#                 help="alpha spr param")
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
                    default=10,
                    help="epochs after pruning")
# parser.add_argument("--threshold",
#                 type=float,
#                 default=5e-2,
#                 help="accuracy drop threshold for pruning")   
# parser.add_argument("--threshold_str",
#                 type=float,
#                 default=5e-3,
#                 help="accuracy drop threshold for pruning")  
# # adversarial problem configuration
# parser.add_argument("--samp_id",
#         type=int,
#         default=10000,
#         help="id of the sample to use for adversarial problem")
parser.add_argument("--time",
                type=float,
                default=180,
                help="mip time limit")
parser.add_argument("--obbt",
                type=int,
                default=-1,
                help="OBBT gurobi parameter")
parser.add_argument("--cuts",
            type=int,
            default=-1,
            help="cuts gurobi parameter")
setting = parser.parse_args()


# config setting
confset = {'arch':['2x40'],
           'reg':['spr'],
           'lamb':[0.0],
           'alpha':[0.0],
           'thr':[5e-2],
           'thr_str':[5e-3],
           'id':[10000+i+1 for i in range(10)],
           'delta':[13]}

setting.save_path = os.path.join(setting.save_path,setting.dataset)


path_bkp = setting.save_path
# setting.prune = True
for arch, reg, lamb, alpha, thr, thr_str, id, delta in itertools.product(*tuple(confset.values())):
    # set config
    setting.arch = arch
    setting.reg = reg
    setting.lamb = lamb
    setting.alpha = alpha
    setting.threshold = thr
    setting.threshold_str = thr_str
    setting.samp_id = id
    setting.delta = delta
    # if arch =='6x500': setting.time = 1800

    print("===================================================================")
    print("===================================================================")
    print()
    print("Experiments configuration:")
    print(setting)
    print()
    run(setting)
    breakpoint()
    setting.save_path = path_bkp
    print("===================================================================")
    print("===================================================================")
    print()
    print()
