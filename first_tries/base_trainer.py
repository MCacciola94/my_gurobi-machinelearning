import torch
class trainer():

    def __init__(self, nn_model, max_epochs=50, lr=0.1):
        self.model = nn_model
        self.max_epochs = max_epochs
        self.lr=lr

    def fit(self, train_loader):
        model = self.model
        optim = torch.optim.SGD(model.parameters(), lr= self.lr,momentum=0)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100], last_epoch= - 1)
        criterion  = torch.nn.CrossEntropyLoss()
        # criterion  = torch.nn.NLLLoss()



        for epoch in range(self.max_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1,28*28)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # backwards
                optim.zero_grad()

                loss.backward()
                optim.step()


                if(i%100) == 0:
                    print(f"epoch {epoch+1}/{self.max_epochs}, loss = {loss.item():.4f}")

    def score(self,test_loader):
        model = self.model
        with torch.no_grad():
            n_correct = 0
            n_samples =0
            for images, labels in test_loader:
                images = images.reshape(-1,28*28)
                outputs = model(images)
                # value, index
                _, predictions = torch.max(outputs,1)
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()
                
            acc = 100 * n_correct / n_samples
        return acc