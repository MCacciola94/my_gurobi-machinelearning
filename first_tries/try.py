import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
from gurobi_ml import add_predictor_constr
from time import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def peak2d(xx, yy):
    return (
        3 * (1 - xx) ** 2.0 * np.exp(-(xx**2) - (yy + 1) ** 2)
        - 10 * (xx / 5 - xx**4 - yy**5) * np.exp(-(xx**2) - yy**2)
        - 1 / 3 * np.exp(-((xx + 1) ** 2) - yy**2)
    )

x = torch.arange(-2, 2, 0.01)
y = torch.arange(-2, 2, 0.01)
x1, x2 = torch.meshgrid(x, y, indexing="ij")
z = peak2d(x1, x2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm, linewidth=0.01, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('peak2d.png')

X = torch.cat([x1.ravel().reshape(-1, 1), x2.ravel().reshape(-1, 1)], axis=1)
y = z.ravel()

hs = 100
# Define a simple sequential network
model1 = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, 1),
)
model2 = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 1),
)
hs =34
model3 = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, hs),
    torch.nn.ReLU(),
    torch.nn.Linear(hs, 1),
)
model=model1
model=model.to(device)
X=X.to(device)
y = y.to(device)

# Construct our loss function and an Optimizer.
criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
start =time()
for t in range(1000):
    # Zero gradients
    optimizer.zero_grad()
    # Forward pass: Compute predicted y by passing x to the model
    pred = model(X)
    # Compute and print loss
    loss = criterion(pred, y.reshape(-1, 1))
    if t % 100 == 0:
        print(f"iteration {t} loss: {loss.item()}")

    if loss.item() < 1e-4:
        break
    # Backward pass in network to compute gradients
    loss.backward()
    # Update weights
    optimizer.step()
else:
    print(f"iteration {t} loss: {loss.item()} time: {time()-start}")

print(f"Final loss: {loss.item()} time: {time()-start}")

X_test = torch.rand((100, 2)) * 2 - 1
X_test = X_test.to(device)
print('R2 score ',metrics.r2_score(peak2d(X_test[:, 0].cpu(), X_test[:, 1].cpu()), model(X_test).cpu().detach().numpy()))
print('max err ', metrics.max_error(peak2d(X_test[:, 0].cpu(), X_test[:, 1].cpu()), model(X_test).cpu().detach().numpy()))
print(model(X).min())

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(
    x1,
    x2,
    model(X).cpu().detach().numpy().reshape(x1.shape),
    cmap=cm.coolwarm,
    linewidth=0.01,
    antialiased=False,
)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('approx.png')

model = model.cpu()

# Start with classical part of the model
m = gp.Model()

x = m.addMVar((1, 2), lb=-2, ub=2, name="x")
y = m.addMVar(1, lb=-GRB.INFINITY, name="y")

m.setObjective(y.sum(), gp.GRB.MINIMIZE)

# Add network trained by pytorch to Gurobi model to predict y from x
nn2gurobi = add_predictor_constr(m, model, x, y)

m.Params.TimeLimit = 20
m.Params.MIPGap = 0.1

m.optimize()

print(x.X)
print(peak2d(x.X[0, 0], x.X[0, 1]))
print(y.X)
print('Tot time ',time()-start)