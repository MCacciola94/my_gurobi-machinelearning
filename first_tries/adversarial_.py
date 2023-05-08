import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import gurobipy as gp

from gurobi_ml import add_predictor_constr

from base_trainer import trainer

max_epochs = 2
batch_size=128
lr =1e-1
hidden = 200
# Get MNIST digit recognition data set
mnist_train = torchvision.datasets.MNIST(root="../Dataset/MNIST", train=True,transform=
        transforms.ToTensor(), download=True)

mnist_test = torchvision.datasets.MNIST(root="../Dataset/MNIST", train=False,transform=
        transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,
                                           shuffle=True,)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,
                                           shuffle=False)

nn_model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden, hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden, 10),
    # torch.nn.Softmax(1),
)
clf = trainer(
    nn_model,
    max_epochs=max_epochs,
    lr=lr
)

clf.fit(train_loader)

print(f"Training score: {clf.score(train_loader):.4}")
print(f"Validation set score: {clf.score(test_loader):.4}")
nn_regression =nn_model.cpu()# torch.nn.Sequential(*nn_model[:-1]).cpu()

imageno = 10000
image = mnist_train.data[imageno, :]
plt.imshow(image, cmap="gray")

flat_img = image.reshape(-1,28*28).type(torch.float32).squeeze(0)/255
ex_prob = nn_regression.forward(flat_img)

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

pred_constr = add_predictor_constr(m, nn_regression, x, y)

pred_constr.print_stats()

m.Params.BestBdStop = 0.0
m.Params.BestObjStop = 0.0
m.optimize()

if m.ObjVal > 0.0:
    plt.imshow(x.X.reshape((28, 28)), cmap="gray")
    x_input = torch.tensor(x.X.reshape(1, -1), dtype=torch.float32)
    label = torch.argmax(nn_model.forward(x_input))
    print(f"Solution is classified as {label}")
else:
    print("No counter example exists in neighborhood.")

