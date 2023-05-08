# MNIST 
# DataLoader Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU Support
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# print(device)

# hyper parameters
input_size = 784 # 28X28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root="../Dataset/MNIST", train=True,
                                           transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="../Dataset/MNIST", train=False,
                                           transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                                           shuffle=True,)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,
                                           shuffle=False)
# examples = iter(train_loader)
# samples, label = examples.next()
# print(samples.shape,label.shape)

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(samples[i][0], cmap='gray')

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, output_size, num_classes):
#         super(NeuralNet,self).__init__()
#         self.l1 = nn.Linear(input_size,hidden_size)
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size,num_classes)
#     def forward(self,x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         return out
    
nn_model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10),
    # torch.nn.Softmax(1),
)
model = nn_model#NeuralNet(input_size, hidden_size,num_classes)
# loss and optmizer
criterion = nn.CrossEntropyLoss()
optmizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# training loop
n_total_steps = len(train_dataset)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 724
        images = images.reshape(-1,28*28) 
        
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backwards
        loss.backward()
        optmizer.step()
        optmizer.zero_grad()
        
        if(i%100) == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps} loss = {loss.item():.4f}")


# test the train
with torch.no_grad():
    n_correct = 0
    n_samples =0
    for images, labels in train_loader:
        images = images.reshape(-1,28*28)
        outputs = model(images)
        # value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct = (predictions == labels).sum().item()
        
    acc = 100 * n_correct / n_samples
    print(f"accuracy = {acc}")

# test
with torch.no_grad():
    n_correct = 0
    n_samples =0
    for images, labels in test_loader:
        images = images.reshape(-1,28*28)
        outputs = model(images)
        # value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct = (predictions == labels).sum().item()
        
    acc = 100 * n_correct / n_samples
    print(f"accuracy = {acc}")