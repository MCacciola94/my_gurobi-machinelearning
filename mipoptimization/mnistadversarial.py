import numpy as np
from matplotlib import pyplot as plt
from time import time

import torch
import torchvision
import torchvision.transforms as transforms

import gurobipy as gp

from gurobi_ml import add_predictor_constr


def adversarial(model, config):
    dataset = torchvision.datasets.MNIST(root="../Dataset/MNIST", train=True, transform=
        transforms.ToTensor(), download=True)

    imageno = config.samp_id
    image = dataset.data[imageno, :]
    flat_img = image.reshape(-1,28*28).type(torch.float32).squeeze(0)/255
    ex_prob = model.forward(flat_img)

    sorted_labels = torch.argsort(ex_prob)
    right_label = sorted_labels[-1]
    wrong_label = sorted_labels[-2]

    image = flat_img.numpy()  # We need numpy converted image


    m = gp.Model()
    delta = 5

    x = m.addMVar(image.shape, lb=0.0, ub=1.0, name="x")
    y = m.addMVar(ex_prob.detach().numpy().shape, lb=-gp.GRB.INFINITY, name="y")

    abs_diff = m.addMVar(image.shape, lb=0, ub=1, name="abs_diff")

    m.setObjective(y[wrong_label] - y[right_label], gp.GRB.MAXIMIZE)

    # Bound on the distance to example in norm-1
    m.addConstr(abs_diff >= x - image)
    m.addConstr(abs_diff >= -x + image)
    m.addConstr(abs_diff.sum() <= delta)

    pred_constr = add_predictor_constr(m, model, x, y)

    pred_constr.print_stats()

    m.Params.BestBdStop = 0.0
    m.Params.BestObjStop = 0.0
    m.setParam('TimeLimit', config.time)
    m.Params.OBBT = config.obbt
    m.Params.cuts = config.cuts
    m.optimize()
    elapsed = m.Runtime
    nodes = m.NodeCount


    if m.ObjVal > 0.0:
        plt.imshow(x.X.reshape((28, 28)), cmap="gray")
        x_input = torch.tensor(x.X.reshape(1, -1), dtype=torch.float32)
        label = torch.argmax(model.forward(x_input))
        print(f"Solution is classified as {label}")
    else:
        print("No counter example exists in neighborhood.")
    
    return elapsed , nodes

