import torch
from torch import nn
from spiking_data import get_numpy_datasets, SpikingNumpyLoader


train_ds, test_ds = [i[0] for i in get_numpy_datasets('shd', 70)]


shd_train_dataloader = SpikingNumpyLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, )
shd_test_dataloader = SpikingNumpyLoader(test_ds, batch_size=16, shuffle=True, num_workers=0, )

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 20),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (batch_data) in enumerate(dataloader):
        X, y = batch_data
        pred = model(torch.Tensor(X))
        loss = loss_fn(pred, torch.Tensor(y).long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (batch_data) in enumerate(dataloader):
        # Compute prediction and loss
            X, y = batch_data
            y = torch.Tensor(y).long()
            pred = model(torch.Tensor(X))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = NeuralNetwork()
learning_rate = 1e-3
batch_size = 16
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(shd_train_dataloader, model, loss_fn, optimizer)
    test_loop(shd_test_dataloader, model, loss_fn)
print("Done!")